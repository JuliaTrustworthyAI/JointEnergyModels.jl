module Samplers

using Distributions
using Flux
using Flux.Optimise: apply!
using ..JointEnergyModels
using ..JointEnergyModels: AbstractSampler

struct ConditionalSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
    niter::Int
end
ConditionalSampler(𝒟x::Distribution, 𝒟y::Distribution, niter::Int) = ConditionalSampler(𝒟x, 𝒟y, niter)

function (sampler::ConditionalSampler)(model, rule::JointEnergyModels.Optimiser, dims::Dims)

    # Setup:
    x = rand(d, dims)
    y = rand(𝒟y)
    f(x) = energy(model, x, y)

    # Training:
    for i in 1:sampler.niter
        Δ = gradient(f, x)[1]
        Δ = apply!(rule, x, Δ)
        x -= Δ
    end

    return x

end

end