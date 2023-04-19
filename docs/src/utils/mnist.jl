function _resize(x; size=(28, 28))
    if n_digits != 28
        img_source = MLDatasets.convert2image(MNIST, x)
        img_rs = imresize(img_source, size)
        x = permutedims(convert(Array{Float32}, Gray.(img_rs)))
    end
    return x
end

function pre_process(x; noise::Float32=0.03f0)
    ϵ = Float32.(randn(size(x)) * noise)
    x = @.(2 * x - 1) .+ ϵ
    return x
end

function load_mnist_data(; nobs::Int=1000, n_digits::Int=1000)
    # Train Set:
    Xtrain, ytrain = MNIST(split=:train)[:]
    end_train = minimum([nobs, size(Xtrain)[end]])
    Xtrain = Xtrain[:, :, 1:end_train]
    Xtrain = mapslices(x -> _resize(x; size=(n_digits, n_digits)), Xtrain, dims=(1, 2))
    ytrain = ytrain[1:end_train]

    # Test Set:
    Xtest, ytest = MNIST(split=:test)[:]
    end_test = minimum([nobs, size(Xtest)[end]])
    Xtest = Xtest[:, :, 1:end_test]
    Xtest = mapslices(x -> _resize(x; size=(n_digits, n_digits)), Xtest, dims=(1, 2))
    ytest = ytest[1:end_test]

    ## One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    ## Validation Set:
    num_val = Int(round(nobs / 10))
    Xtrain, Xval = (Xtrain[:, :, 1:(end-num_val)], Xtrain[:, :, (end-num_val+1):end])
    Xtrain = mapslices(x -> pre_process(x), Xtrain, dims=(1, 2))
    Xval = mapslices(x -> pre_process(x, noise=0.0f0), Xval, dims=(1, 2))
    ytrain, yval = (ytrain[:, 1:(end-num_val)], ytrain[:, (end-num_val+1):end])

    return Xtrain, ytrain, Xval, yval, Xtest, ytest
end

function samples_real(model::JointEnergyModel, dl::DataLoader, n::Int=16; img_size=n_digits * 5)
    x = reduce((x, y) -> cat(x, y[1], dims=ndims(x)), dl, init=[])
    num_x = Int(round(ceil(sqrt(n))))
    num_y = Int(round(floor(sqrt(n))))
    plot_data = [heatmap(rotl90(x[:, :, rand(1:size(x)[end])]), axis=nothing, cb=false) for i in 1:n]
    plot(plot_data..., layout=(num_x, num_y), size=(num_x * img_size, num_y * img_size), margin=(round(0.05 * img_size), :px))
end

function samples_generated(model::JointEnergyModel, dl::DataLoader, n::Int=16; img_size=n_digits * 5)
    x = reduce((x, y) -> cat(x, y[1], dims=ndims(x)), dl, init=[])
    x = jem.sampler(jem.chain, jem.sampling_rule, size(x))
    num_x = Int(round(ceil(sqrt(n))))
    num_y = Int(round(floor(sqrt(n))))
    plot_data = [heatmap(rotl90(x[:, :, i]), axis=nothing, cb=false) for i in 1:n]
    plot(plot_data..., layout=(num_x, num_y), size=(num_x * img_size, num_y * img_size), margin=(round(0.05 * img_size), :px))
end
