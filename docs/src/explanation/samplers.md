# MNIST


``` julia
using TaijaData: load_mnist
using CounterfactualExplanations.Models: load_mnist_mlp, load_mnist_ensemble
```

``` julia
X, y = load_mnist()
K = size(y, 1)
D = size(X, 1)
mlp = load_mnist_mlp().model
f_mlp(x) = mlp(x)
ens = load_mnist_ensemble().model
f_ens(x) = sum(map(mlp -> mlp(x), ens))/length(ens)
batch_size = 100
```

## Sampling

``` julia
𝒟x = Uniform(0,1)
𝒟y = Categorical(ones(K) ./ K)
sampler = UnconditionalSampler(𝒟x; input_size=(D,))
conditional_sampler = ConditionalSampler(𝒟x, 𝒟y; input_size=(D,))
opt = ImproperSGLD()
n_iter = 256
```

### Conditional Draws

``` julia
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = conditional_sampler(f_mlp, opt; niter=n_iter, y=i, n_samples=neach)
    plts_i = []
    for j in 1:size(x,2)
        xj = reshape(x[:,j], (28,28))
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
    x = conditional_sampler(f_ens, opt; niter=n_iter, y=i, n_samples=neach)
    plts_i = []
    for j in 1:size(x,2)
        xj = reshape(x[:,j], (28,28))
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```

### Unconditional Draws

``` julia
_w = 1500
plts = []
neach = 10
for i in 1:10
    x = sampler(f_mlp, opt; niter=n_iter, n_samples=neach)
    plts_i = []
    for j in 1:size(x,2)
        xj = reshape(x[:,j], (28,28))
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
    x = sampler(f_ens, opt; niter=n_iter, n_samples=neach)
    plts_i = []
    for j in 1:size(x,2)
        xj = reshape(x[:,j], (28,28))
        plts_i = [plts_i..., heatmap(rotl90(xj), axis=nothing, cb=false)]
    end
    plt = plot(plts_i..., size=(_w,0.10*_w), layout=(1,10))
    plts = [plts..., plt]
end
plot(plts..., size=(_w,_w), layout=(10,1))
```
