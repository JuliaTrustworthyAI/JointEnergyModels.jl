using ChainRulesCore
using Flux
using Flux.Losses: logitcrossentropy
using EnergySamplers: ImproperSGLD

"Base class for joint energy models."
struct JointEnergyModel
    chain::Chain
    sampler::AbstractSampler
    sampling_rule::AbstractSamplingRule
    sampling_steps::Int
end

"""
    JointEnergyModel(
        chain::Union{Chain,Nothing},
        sampler::AbstractSampler;
        sampling_rule = ImproperSGLD(),
        sampling_steps = sampling_rule isa ImproperSGLD ? 10 : 1000,
    )

Constructs a `JointEnergyModel` object. The `JointEnergyModel` object is a wrapper around a `Chain` model and a `Sampler` object. The `Sampler` object is used to generate samples from the model's energy function. The `sampling_rule` and `sampling_steps` parameters are used to specify the sampling rule and the number of sampling steps, respectively.

# Arguments

- `chain::Union{Chain,Nothing}`: The `Chain` model.
- `sampler::AbstractSampler`: The `Sampler` object.
- `sampling_rule::AbstractSamplingRule`: The sampling rule to use. Default is `ImproperSGLD()`.
- `sampling_steps::Int`: The number of sampling steps.

# Returns

- `jem::JointEnergyModel`: The `JointEnergyModel` object.
"""
function JointEnergyModel(
    chain::Union{Chain,Nothing},
    sampler::AbstractSampler;
    sampling_rule = ImproperSGLD(),
    sampling_steps = sampling_rule isa ImproperSGLD ? 10 : 1000,
)
    JointEnergyModel(chain, sampler, sampling_rule, sampling_steps)
end

Flux.@functor JointEnergyModel

"""
    (jem::JointEnergyModel)(x)

Computes the output of the joint energy model.
"""
function (jem::JointEnergyModel)(x)
    jem.chain(x)
end

@doc raw"""
    class_loss(jem::JointEnergyModel, x, y)

Computes the classification loss. The (default) classification loss is the cross-entropy loss between the predicted and target labels. The loss is aggregated using the `agg` function.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `x`: The input data.
- `y`: The target data.
- `loss_fun`: The loss function to use.
- `agg`: The aggregation function to use for the loss.

# Returns

- `ℓ`: The classification loss.
"""
function class_loss(jem::JointEnergyModel, x, y; loss_fun = logitcrossentropy, agg = mean)
    ŷ = jem(x)
    ℓ = loss_fun(ŷ, y, agg = agg)
    ℓ = ℓ isa Matrix ? vec(ℓ) : ℓ
    return ℓ
end

"""
    get_samples(jem::JointEnergyModel, x)::Tuple{AbstractArray,AbstractArray}

Gets samples from the sampler buffer. The number of samples is determined by the size of the input data `x` and the buffer. If the batch of input data is larger than the buffer, a subset of the input data is sampled.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `x`: The input data.

# Returns

- `x`: The input data.
- `xsample`: The samples from the buffer.
"""
function get_samples(jem::JointEnergyModel, x)
    # Determine the size of the batch:
    # Either the size of the input data (training batch size) or the total size of the buffer, whichever is smaller.
    size_sample =
        minimum([size(x)[end], size(jem.sampler.buffer, ndims(jem.sampler.buffer))])
    # If the input batch is larger than the buffer, we need to sample a subset of the input data.
    if size_sample < size(x)[end]
        x = selectdim(x, ndims(x), rand(1:size(x)[end], size_sample))
    end
    # Get the `size_sample` samples from the buffer that were last added:
    xsample = selectdim(jem.sampler.buffer, ndims(jem.sampler.buffer), 1:size_sample)
    @assert size(xsample) == size(x)
    return x, xsample
end

@doc raw"""
    gen_loss(jem::JointEnergyModel, x)

Computes the generative loss. The generative loss is the difference between the energy of the input data and the energy of the generated samples from the replay buffer.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `x`: The input data.

# Returns

- `ℓ`: The generative loss, which is the difference between the energy of the input data and the energy of the generated samples from the replay buffer.
"""
function gen_loss(jem::JointEnergyModel, x, y)
    x, xsample = get_samples(jem, x)
    E(x) = energy(jem.sampler, jem.chain, x, onecold(y)[1])
    ℓ = E(x) .- E(xsample)
    return ℓ
end

@doc raw"""
    reg_loss(jem::JointEnergyModel, x)

Computes the regularization loss. The regularization loss is the sum of the squared energies of the input data and the generated samples from the replay buffer. This loss is used to prevent the model from overfitting with respect to the generative loss.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `x`: The input data.
- `y`: The target data.

# Returns

- `ℓ`: The regularization loss, which is the sum of the squared energies of the input data and the generated samples from the replay buffer.
"""
function reg_loss(jem::JointEnergyModel, x, y)
    x, xsample = get_samples(jem, x)
    E(x) = energy(jem.sampler, jem.chain, x, onecold(y)[1])
    ℓ = E(x) .^ 2 .+ E(xsample) .^ 2
    return ℓ
end

@doc raw"""
    loss(jem::JointEnergyModel, x, y; agg=mean)

Computes the total loss. The total loss is a weighted sum of the classification, generative, and regularization losses. The weights are determined by the `α` parameter.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `x`: The input data.
- `y`: The target data.
- `agg`: The aggregation function to use for the loss.
- `α`: The weights for the classification, generative, and regularization losses.
- `use_class_loss`: Whether to use the classification loss.
- `use_gen_loss`: Whether to use the generative loss.
- `use_reg_loss`: Whether to use the regularization loss.
- `class_loss_fun`: The classification loss function to use.

# Returns

- `loss`: The total loss.
"""
function loss(
    jem::JointEnergyModel,
    x,
    y;
    agg = mean,
    α = [1.0, 1.0, 0.01],
    use_class_loss::Bool = true,
    use_gen_loss::Bool = true,
    use_reg_loss::Bool = true,
    class_loss_fun::Function = logitcrossentropy,
)

    if use_gen_loss || use_reg_loss
        xsample = []
        Flux.testmode!(jem.chain)
        ignore_derivatives() do
            _xsample = jem.sampler(jem.chain, jem.sampling_rule; niter = jem.sampling_steps)
            push!(xsample, _xsample)
        end
        Flux.trainmode!(jem.chain)
    end

    ℓ_clf = use_class_loss ? class_loss(jem, x, y; loss_fun = class_loss_fun) : 0.0
    ℓ_gen = use_gen_loss ? gen_loss(jem, x, y) : 0.0
    ℓ_reg = use_reg_loss ? reg_loss(jem, x, y) : 0.0
    loss = agg(α[1] * ℓ_clf .+ α[2] * ℓ_gen .+ α[3] * ℓ_reg)
    return loss
end

"""
    generate_samples(jem::JointEnergyModel, n::Int; kwargs...)

A convenience function for generating samples for a given energy model. If `n` is `missing`, then the sampler's `batch_size` is used. The `kwargs` are passed to the sampler when it is called.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `n::Int`: The number of samples to generate.
- `kwargs`: Additional keyword arguments to pass to the sampler when it is called.

# Returns

- `samples`: The generated samples.
"""
function generate_samples(jem::JointEnergyModel, n::Int; kwargs...)
    n = ismissing(n) ? nothing : n
    sampler = jem.sampler
    model = jem.chain
    rule = jem.sampling_rule
    samples = sampler(model, rule; n_samples = n, kwargs...)
    return samples
end

"""
    generate_conditional_samples(jem::JointEnergyModel, n::Int, y::Int; kwargs...)

A convenience function for generating conditional samples for a given model, sampler and sampling rule. If `n` is `missing`, then the sampler's `batch_size` is used. The conditioning value `y` needs to be specified.

# Arguments

- `jem::JointEnergyModel`: The joint energy model.
- `n::Int`: The number of samples to generate.
- `y::Int`: The conditioning value.
- `kwargs`: Additional keyword arguments to pass to the sampler when it is called.

# Returns

- `samples`: The generated samples.
"""
function generate_conditional_samples(jem::JointEnergyModel, n::Int, y::Int; kwargs...)
    @assert typeof(jem.sampler) <: ConditionalSampler "sampler must be a ConditionalSampler"
    return generate_samples(jem, n; kwargs..., y = y)
end
