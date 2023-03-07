module Samplers

using Distributions
using Flux
using Flux.Optimise: apply!
using ..JointEnergyModels
using ..JointEnergyModels: AbstractSampler

export ConditionalSampler, UnconditionalSampler

@doc raw"""
    ConditionalSampler <: AbstractSampler

Generates conditional samples: $x \sim p(x|y).$
"""
struct ConditionalSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
end

function (sampler::ConditionalSampler)(
    model, rule::JointEnergyModels.AbstractSamplingRule, dims::Dims;
    niter::Int=100, y::Union{Nothing,Int}=nothing
)


    # Setup:
    x = Float32.(rand(sampler.𝒟x, dims...))
    if isnothing(y)
        y = rand(sampler.𝒟y)
    end
    f(x) = energy(model, x, y)

    # Training:
    for i in 1:niter
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
    𝒟x::Distribution
end

function (sampler::UnconditionalSampler)(
    model, rule::JointEnergyModels.AbstractSamplingRule, dims::Dims;
    niter::Int=20
)

    # Setup:
    x = Float32.(rand(sampler.𝒟x, dims...))
    f(x) = energy(model, x)

    # Training:
    for i in 1:niter
        Δ = gradient(f, x)[1]
        Δ = apply!(rule, x, Δ)
        x -= Δ
    end

    return x

end

end