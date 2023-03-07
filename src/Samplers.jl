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
ConditionalSampler(; 𝒟x::Distribution, 𝒟y::Distribution) = ConditionalSampler(𝒟x, 𝒟y)

function (sampler::ConditionalSampler)(
    model, rule::JointEnergyModels.AbstractOptimiser, input_dim::Int, batchsize::Int=1; 
    y::Union{Nothing, Int}=nothing, niter::Int=100, as_matrix::Bool=true
)

    # Setup:
    x = map(i -> Float32.(rand(sampler.𝒟x, input_dim, 1)), 1:batchsize)
    if isnothing(y)
        y = rand(sampler.y)
    end
    f(x) = energy(model, x, y)

    # Training:
    x = map(x) do _x
        for i in 1:niter
            Δ = gradient(f, _x)[1]
            Δ = apply!(rule, _x, Δ)
            _x -= Δ
        end
        return _x
    end

    if as_matrix
        x = reduce(hcat, x)
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
UnconditionalSampler(; 𝒟x::Distribution) = UnconditionalSampler(𝒟x)

function (sampler::UnconditionalSampler)(
    model, rule::JointEnergyModels.AbstractOptimiser, input_dim::Int, batchsize::Int=1;
    niter::Int=100, as_matrix::Bool=true
)

    # Setup:
    x = map(i -> Float32.(rand(sampler.𝒟x, input_dim, 1)), 1:batchsize)
    f(x) = energy(model, x)

    # Training:
    x = map(x) do _x
        for i in 1:niter
            Δ = gradient(f, _x)[1]
            Δ = apply!(rule, _x, Δ)
            _x -= Δ
        end
        return _x
    end

    if as_matrix
        x = reduce(hcat, x)
    end

    return x

end

end