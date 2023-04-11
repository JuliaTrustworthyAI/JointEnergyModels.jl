using ChainRulesCore
using Flux
using Flux: logsumexp
using Flux.Losses: logitcrossentropy

struct JointEnergyModel
    chain::Chain
    sampler::AbstractSampler
    sampling_rule::AbstractSamplingRule
    sampling_steps::Int
end

function JointEnergyModel(
    chain::Chain, sampler::AbstractSampler;
    sampling_rule=ImproperSGLD(),
    sampling_steps=10
)
    JointEnergyModel(chain, sampler, sampling_rule, sampling_steps)
end

Flux.@functor JointEnergyModel

function (jem::JointEnergyModel)(x)
    jem.chain(x)
end

@doc raw"""
    class_loss(jem::JointEnergyModel, x, y)

Computes the classification loss.
"""
function class_loss(jem::JointEnergyModel, x, y)
    ŷ = jem(x)
    ℓ = logitcrossentropy(ŷ, y; agg=x -> x)
    return ℓ
end

@doc raw"""
    gen_loss(jem::JointEnergyModel, x)

Computes the generative loss.
"""
function gen_loss(jem::JointEnergyModel, x)
    ŷ = jem(x)
    xsample = []
    ignore_derivatives() do
        _xsample = jem.sampler(jem.chain, jem.sampling_rule; niter=jem.sampling_steps, n_samples=size(ŷ)[2])
        push!(xsample, _xsample)
    end
    ŷsample = jem(xsample...)
    ℓ = logsumexp(ŷ; dims=1) .- logsumexp(ŷsample; dims=1)
    return ℓ
end

function reg_loss(jem::JointEnergyModel, x)
    ŷ = jem(x)
    xsample = []
    ignore_derivatives() do
        _xsample = jem.sampler(jem.chain, jem.sampling_rule; niter=jem.sampling_steps, n_samples=size(ŷ)[2])
        push!(xsample, _xsample)
    end
    ŷsample = jem(xsample...)
    ℓ = logsumexp(ŷ; dims=1) .^ 2 .+ logsumexp(ŷsample; dims=1) .^ 2
    return ℓ
end

@doc raw"""
    loss(jem::JointEnergyModel, x, y; agg=mean)

Computes the total loss.
"""
function loss(jem::JointEnergyModel, x, y; agg=mean, α=0.1)
    ℓ_clf = class_loss(jem, x, y)
    ℓ_gen = gen_loss(jem, x)
    ℓ_reg = reg_loss(jem, x)
    loss = agg(ℓ_clf .+ ℓ_gen .+ α * ℓ_reg)
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
    return (sampler::AbstractSampler)(model, rule; n_samples=n, kwargs...)
end

"""
    generate_conditional_samples(model, rule::Flux.Optimise.AbstractOptimiser, n::Int, y::Int; kwargs...)

A convenience function for generating conditional samples for a given model, sampler and sampling rule. If `n` is `missing`, then the sampler's `batch_size` is used. The conditioning value `y` needs to be specified.
"""
function generate_conditional_samples(jem::JointEnergyModel, n::Int, y::Int; kwargs...)
    @assert typeof(jem.sampler) <: ConditionalSampler "sampler must be a ConditionalSampler"
    return generate_samples(jem, n; kwargs..., y=y)
end