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
    sampling_steps=10,
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

@doc raw"""
    loss(jem::JointEnergyModel, x, y; agg=mean)

Computes the total loss.
"""
function loss(jem::JointEnergyModel, x, y; agg=mean)
    ℓ_clf = class_loss(jem, x, y)
    ℓ_gen = gen_loss(jem, x)
    loss = agg(ℓ_clf .+ ℓ_gen)
    return loss
end