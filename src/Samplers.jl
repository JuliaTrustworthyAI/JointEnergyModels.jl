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
    model, rule::JointEnergyModels.AbstractSamplingRule, input_dims::Dims;
    batchsize::Int=1, y::Union{Nothing, Int}=nothing, niter::Int=100, as_array::Bool=true
)

    # Setup:
    x = map(i -> Float32.(rand(sampler.𝒟x, input_dims..., 1)), 1:batchsize)
    if isnothing(y)
        y = rand(sampler.𝒟y)
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

    if as_array
        x = reduce((x1, x2) -> cat(x1,x2,dims=length(input_dims) + 1), x)
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
    model, rule::JointEnergyModels.AbstractSamplingRule, input_dims::Dims;
    batchsize::Int=1, niter::Int=20, as_array::Bool=true
)

    # Setup:
    x = map(i -> Float32.(rand(sampler.𝒟x, input_dims..., 1)), 1:batchsize)
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

    if as_array
        x = reduce((x1, x2) -> cat(x1, x2, dims=length(input_dims) + 1), x)
    end

    return x

end

end