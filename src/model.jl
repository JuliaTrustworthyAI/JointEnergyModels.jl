using Flux
using Flux: logsumexp
using Flux.Losses: logitcrossentropy

struct JointEnergyModel
    chain::Chain
    sampler::AbstractSampler
    sampling_rule::AbstractSamplingRule
end

function JointEnergyModel(chain::Chain, sampler::AbstractSampler; sampling_rule=ImproperSGLD())
    JointEnergyModel(chain, sampler, sampling_rule)
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
    xsample = jem.sampler(jem.chain, jem.sampling_rule, size(x)[1:end-1]; batchsize=size(x)[end])
    ŷsample = jem(xsample)
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