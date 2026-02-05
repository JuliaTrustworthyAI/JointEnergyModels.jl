using ChainRulesCore
using Flux
using Flux.Losses: logitcrossentropy
using EnergySamplers: ImproperSGLD

struct JointEnergyModel
    chain::Chain
    sampler::AbstractSampler
    sampling_rule::AbstractSamplingRule
    sampling_steps::Int
end

function JointEnergyModel(
    chain::Union{Chain,Nothing},
    sampler::AbstractSampler;
    sampling_rule=ImproperSGLD(),
    sampling_steps=sampling_rule isa ImproperSGLD ? 10 : 1000,
)
    JointEnergyModel(chain, sampler, sampling_rule, sampling_steps)
end

function (jem::JointEnergyModel)(x)
    jem.chain(x)
end

@doc raw"""
    class_loss(jem::JointEnergyModel, x, y)

Computes the classification loss.
"""
function class_loss(nn, jem::JointEnergyModel, x, y; loss_fun=logitcrossentropy, agg=mean)
    ŷ = jem(x)
    ℓ = loss_fun(ŷ, y, agg=agg)
    ℓ = ℓ isa Matrix ? vec(ℓ) : ℓ
    return ℓ
end

"""
    get_samples(jem::JointEnergyModel, x)

Gets samples from the sampler buffer.
"""
function get_samples(jem::JointEnergyModel, x)
    size_sample =
        minimum([size(x)[end], size(jem.sampler.buffer, ndims(jem.sampler.buffer))])
    if size_sample < size(x)[end]
        x = selectdim(x, ndims(x), rand(1:size(x)[end], size_sample))
    end
    xsample = selectdim(jem.sampler.buffer, ndims(jem.sampler.buffer), 1:size_sample)
    @assert size(xsample) == size(x)
    return x, xsample
end

@doc raw"""
    gen_loss(jem::JointEnergyModel, x)

Computes the generative loss.
"""
function gen_loss(nn, jem::JointEnergyModel, x, y)
    # Training batch `x` and generated samples `xsample`:
    x, xsample = get_samples(jem, x)
    E(x) = energy(jem.sampler, nn, x, onecold(y)[1])

    # E(observed) - E(generated):
    ℓ = E(x) .- E(xsample)
    return ℓ
end

@doc raw"""
    reg_loss(jem::JointEnergyModel, x)

Computes the regularization loss.
"""
function reg_loss(nn, jem::JointEnergyModel, x, y)
    x, xsample = get_samples(jem, x)
    E(x) = energy(jem.sampler, nn, x, onecold(y)[1])
    ℓ = E(x) .^ 2 .+ E(xsample) .^ 2
    return ℓ
end

@doc raw"""
    loss(jem::JointEnergyModel, x, y; agg=mean)

Computes the total loss.
"""
function loss(
    nn,
    jem::JointEnergyModel,
    x,
    y;
    agg=mean,
    α=[1.0, 1.0, 0.01],
    use_class_loss::Bool=true,
    use_gen_loss::Bool=true,
    use_reg_loss::Bool=true,
    class_loss_fun::Function=logitcrossentropy,
)

    if use_gen_loss || use_reg_loss
        xsample = []
        Flux.testmode!(nn)
        ignore_derivatives() do
            _xsample = jem.sampler(nn, jem.sampling_rule; niter=jem.sampling_steps)
            push!(xsample, _xsample)
        end
        Flux.trainmode!(nn)
    end

    ℓ_clf = use_class_loss ? class_loss(nn, jem, x, y; loss_fun=class_loss_fun) : 0.0
    ℓ_gen = use_gen_loss ? gen_loss(nn, jem, x, y) : 0.0
    ℓ_reg = use_reg_loss ? reg_loss(nn, jem, x, y) : 0.0
    loss = agg(α[1] * ℓ_clf .+ α[2] * ℓ_gen .+ α[3] * ℓ_reg)
    return loss
end

"""
    generate_samples(jem::JointEnergyModel, n::Int; kwargs...)

A convenience function for generating samples for a given energy model. If `n` is `missing`, then the sampler's `batch_size` is used.
"""
function generate_samples(jem::JointEnergyModel, n::Int; kwargs...)
    n = ismissing(n) ? nothing : n
    sampler = jem.sampler
    model = jem.chain
    rule = jem.sampling_rule
    samples = sampler(model, rule; n_samples=n, kwargs...)
    return samples
end

"""
    generate_conditional_samples(model, rule::Flux.Optimise.AbstractOptimiser, n::Int, y::Int; kwargs...)

A convenience function for generating conditional samples for a given model, sampler and sampling rule. If `n` is `missing`, then the sampler's `batch_size` is used. The conditioning value `y` needs to be specified.
"""
function generate_conditional_samples(jem::JointEnergyModel, n::Int, y::Int; kwargs...)
    @assert typeof(jem.sampler) <: ConditionalSampler "sampler must be a ConditionalSampler"
    return generate_samples(jem, n; kwargs..., y=y)
end
