module Samplers

using Distributions
using Flux
using Flux.Optimise: apply!
using ..JointEnergyModels
using ..JointEnergyModels: AbstractSampler

@doc raw"""
    ConditionalSampler <: AbstractSampler

Generates conditional samples: $x \sim p(x|y).$
"""
struct ConditionalSampler <: AbstractSampler
    𝒟ₓ::Distribution
    y::Union{Int,Distribution}
    niter::Int
end
ConditionalSampler(𝒟ₓ::Distribution, y::Union{Int,Distribution}, niter::Int) = ConditionalSampler(𝒟ₓ, y, niter)

function (sampler::ConditionalSampler)(model, rule::JointEnergyModels.Optimiser, dims::Dims)

    # Setup:
    x = rand(sampler.𝒟ₓ, dims)
    if typeof(y) <: Distribution
        y = rand(sampler.y)
    end
    f(x) = energy(model, x, y)

    # Training:
    for i in 1:sampler.niter
        Δ = gradient(f, x)[1]
        Δ = apply!(rule, x, Δ)
        x -= Δ
    end

    return x

end

@doc raw"""
    UnonditionalSampler <: AbstractSampler

Generates unconditional samples: $x \sim p(x).$
"""
struct UnconditionalSampler <: AbstractSampler
    𝒟ₓ::Distribution
    niter::Int
end
UnconditionalSampler(𝒟ₓ::Distribution, niter::Int) = UnconditionalSampler(𝒟ₓ, niter)

function (sampler::UnconditionalSampler)(model, rule::JointEnergyModels.Optimiser, dims::Dims)

    # Setup:
    x = rand(sampler.𝒟ₓ, dims)
    f(x) = energy(model, x)

    # Training:
    for i in 1:sampler.niter
        Δ = gradient(f, x)[1]
        Δ = apply!(rule, x, Δ)
        x -= Δ
    end

    return x

end

end