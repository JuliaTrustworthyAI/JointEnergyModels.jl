# Compatibility with `MLJFlux`


## Synthetic Data

``` julia
nobs=2000
X, y = make_circles(nobs, noise=0.1, factor=0.5)
Xplot = Float32.(permutedims(matrix(X)))
X = table(permutedims(Xplot))
display(scatter(Xplot[1,:], Xplot[2,:], group=y, label=""))
batch_size = Int(round(nobs/10))
```

``` julia
sampler = ConditionalSampler(X, y, batch_size=batch_size)
clf = JointEnergyClassifier(
    sampler;
    builder=MLJFlux.MLP(hidden=(32, 32, 32,), σ=Flux.relu),
    batch_size=batch_size,
    finaliser=Flux.softmax,
    loss=Flux.Losses.crossentropy,
    jem_training_params=(α=[1.0,1.0,1e-1],verbosity=10,),
)
```

``` julia
println(typeof(clf) <: MLJFlux.MLJFluxModel)
```

``` julia
mach = machine(clf, X, y)
fit!(mach)
```

``` julia
jem = mach.model.jem
batch_size = mach.model.batch_size
X = Float32.(permutedims(matrix(X)))
y_labels = Int.(y.refs)
y = Flux.onehotbatch(y.refs, sort(unique(y_labels)))
```

``` julia
if typeof(jem.sampler) <: ConditionalSampler
    
    plts = []
    for target in 1:size(y,1)
        X̂ = generate_conditional_samples(jem, batch_size, target; niter=1000) 
        ex = extrema(hcat(X,X̂), dims=2)
        xlims = ex[1]
        ylims = ex[2]
        x1 = range(1.0f0.*xlims...,length=100)
        x2 = range(1.0f0.*ylims...,length=100)
        plt = contour(
            x1, x2, (x, y) -> softmax(jem([x, y]))[target], 
            fill=true, alpha=0.5, title="Target: $target", cbar=false,
            xlims=xlims,
            ylims=ylims,
        )
        scatter!(X[1,:], X[2,:], color=vec(y_labels), group=vec(y_labels), alpha=0.5)
        scatter!(
            X̂[1,:], X̂[2,:], 
            color=repeat([target], size(X̂,2)), 
            group=repeat([target], size(X̂,2)), 
            shape=:star5, ms=10
        )
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(1, size(y,1)), size=(size(y,1)*400, 400))
    display(plt)
end
```

## MNIST

``` julia
# Data:
nobs = 1000
n_digits = 28
Xtrain, ytrain, _, _, _, _ = load_mnist_data(nobs=nobs, n_digits=n_digits)
Xtrain = table(permutedims(MLUtils.flatten(Xtrain)))
ytrain = coerce(Flux.onecold(ytrain, 0:9), Multiclass)

# Hyperparameters:
D = n_digits^2             
K = 10                      
M = 32
lr = 1e-3           
num_epochs = 500
max_patience = 5            
batch_size = Int(round(nobs/10))
α = [1.0,1.0,1e-2]
```

``` julia
activation = Flux.swish
builder = MLJFlux.MLP(hidden=(M,M,M,), σ=activation)
```

``` julia
# We initialize the full model
𝒟x = Uniform(-1,1)
𝒟y = Categorical(ones(K) ./ K)
sampler = ConditionalSampler(𝒟x, 𝒟y, input_size=(D,), batch_size=10)
clf = JointEnergyClassifier(
    sampler;
    builder=builder,
    batch_size=batch_size,
    finaliser=Flux.softmax,
    loss=Flux.Losses.crossentropy,
    jem_training_params=(α=α,verbosity=10,),
    sampling_steps=20,
    optimiser=Flux.Optimise.Adam(lr),
)
```

``` julia
mach = machine(clf, Xtrain, ytrain)
fit!(mach)
```

``` julia
jem = mach.model.jem
n_iter = 1000
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = jem.sampler(jem.chain, jem.sampling_rule; niter=n_iter, n_samples=neach, y=i)
    plts_i = []
    for j in 1:size(x, 2)
        xj = x[:,j]
        xj = reshape(xj, (n_digits, n_digits))
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```
