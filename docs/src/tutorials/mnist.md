# Joint Energy Models


## Data

``` julia
nobs = 1000
n_digits = 28
Xtrain, ytrain, Xval, yval, Xtest, ytest = load_mnist_data(nobs=nobs, n_digits=n_digits)
```

## `JointEnergyModel`

### Hyperparameters

``` julia
D = n_digits               
K = 10                      
M = 128
lr = 1e-3                
num_epochs = 100
max_patience = 5            
batchsize = Int(round(nobs/10))
```

### Initializing the model

``` julia
activation = relu
mlp = Chain(
    MLUtils.flatten,
    Dense(prod((D,D)), M, activation),
    # BatchNorm(M, activation),
    # Dense(M, M),
    # BatchNorm(M, activation),
    Dense(M, K),
)

# We initialize the full model
𝒟x = Uniform(-1,1)
𝒟y = Categorical(ones(K) ./ K)
sampler = ConditionalSampler(𝒟x, 𝒟y, input_size=(D,D), batch_size=10)
jem = JointEnergyModel(
    mlp, sampler;
    sampling_steps=20,
)
```

### Training loop

``` julia
# Initialise 
opt = Adam(lr)
opt_state = Flux.setup(opt, jem)
train_set = DataLoader((Xtrain, ytrain); batchsize=batchsize, shuffle=true)
val_set = DataLoader((Xval, yval); batchsize=batchsize, shuffle=false)
test_set = DataLoader((Xtest, ytest); batchsize=batchsize, shuffle=false)
```

``` julia
logs = train_model(
    jem, train_set, opt_state; num_epochs=num_epochs, val_set=val_set,
    verbosity = minimum([num_epochs, 10]),
    α = [1.0,1.0,1e-2],
    # use_class_loss=false,
    # use_gen_loss=false,
    # use_reg_loss=false,
)
```

### The final evaluation

``` julia
n_iter = 200
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = jem.sampler(jem.chain, jem.sampling_rule; niter=n_iter, n_samples=neach, y=i)
    plts_i = []
    for j in 1:size(x, 3)
        xj = x[:,:,j]
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```

#### From Scratch

``` julia
sampler = UnconditionalSampler(𝒟x; input_size=(D,D))
conditional_sampler = ConditionalSampler(𝒟x, 𝒟y; input_size=(D,D))
opt = ImproperSGLD(10.0,0.005)
n_iter = 256
```

``` julia
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = conditional_sampler(jem.chain, opt; niter=n_iter, y=i, n_samples=neach)
    plts_i = []
    for j in 1:size(x,3)
        xj = x[:,:,j]
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```

``` julia
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = sampler(jem.chain, opt; niter=n_iter, y=i, n_samples=neach)
    plts_i = []
    for j in 1:size(x,3)
        xj = x[:,:,j]
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```
