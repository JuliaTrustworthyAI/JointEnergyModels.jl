module Samplers

using Distributions
using Flux
using Flux.Optimise: apply!, Optimiser
using ..JointEnergyModels
using ..JointEnergyModels: AbstractSampler
using MLUtils
using StatsBase

export ConditionalSampler, UnconditionalSampler
export energy

include("pre_processing.jl")

"""
    (sampler::AbstractSampler)(
    model, rule::Flux.Optimise.AbstractOptimiser;
    kwargs...
)

Base method for generating samples for a given models, sampler and sampling rule.
"""
function (sampler::AbstractSampler)(
    model, rule::Flux.Optimise.AbstractOptimiser;
    niter::Int=100,
    clip_grads::Union{Nothing,AbstractFloat}=1e-2,
    n_samples::Union{Nothing,Int}=nothing,
    kwargs...
)

    n_samples = isnothing(n_samples) ? sampler.batch_size : n_samples
    inp_samples = Float32.(rand(sampler.𝒟x, sampler.input_size..., n_samples))

    # # Choose 95% of the batch from the buffer, 5% generate from scratch:
    n_new = rand(Binomial(n_samples, (1.0 - sampler.prob_buffer)))
    rand_imgs = Float32.(rand(sampler.𝒟x, sampler.input_size..., n_new))
    old_imgs = selectdim(sampler.buffer, ndims(sampler.buffer), rand(1:size(sampler.buffer, ndims(sampler.buffer)), n_samples - n_new))
    inp_samples = Float32.(cat(rand_imgs, old_imgs, dims=ndims(sampler.buffer)))

    # Perform MCMC sampling:
    rule = isnothing(clip_grads) ? rule : Optimiser(ClipValue(clip_grads), rule)
    Flux.testmode!(model)
    inp_samples = mcmc_samples(
        sampler, model, rule, inp_samples;
        niter=niter,
        kwargs...
    )
    Flux.trainmode!(model)
    inp_samples = Float32.(clamp.(inp_samples, minimum(sampler.𝒟x), maximum(sampler.𝒟x)))

    # Update buffer:
    sampler.buffer = cat(inp_samples, sampler.buffer, dims=ndims(sampler.buffer))
    _end = minimum([size(sampler.buffer, ndims(sampler.buffer)), sampler.max_len])
    sampler.buffer = selectdim(sampler.buffer, ndims(sampler.buffer), 1:_end)

    return inp_samples

end

@doc raw"""
    ConditionalSampler <: AbstractSampler

Generates conditional samples: $x \sim p(x|y).$
"""
mutable struct ConditionalSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
    transformer::Union{Nothing,AbstractDataTransform}
end

"""
    ConditionalSampler(
        𝒟x::Distribution, 𝒟y::Distribution;
        input_size::Dims, batch_size::Int,
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `ConditionalSampler`.
"""
function ConditionalSampler(
    𝒟x::Distribution, 𝒟y::Distribution;
    input_size::Dims, batch_size::Int=1,
    max_len::Int=10000, prob_buffer::AbstractFloat=0.95
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return ConditionalSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer, nothing)
end

"""
    ConditionalSampler(
        X::AbstractArray, y::AbstractArray;
        batch_size::Int,
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `ConditionalSampler`.
"""
function ConditionalSampler(
    X::AbstractArray, y::AbstractArray;
    batch_size::Int=1,
    max_len::Int=10000, prob_buffer::AbstractFloat=0.95
)
    @assert batch_size <= max_len "batch_size must be <= max_len"

    # Fit data transformer:
    dt = fit(UnitRangeTransform, X, dims=ndims(X))

    # Prior distributions:
    𝒟x = Uniform(-1, 1)                 # following the convention in the literature
    𝒟y = Categorical(1:size(y, 1))      # TODO: make more general

    # Input dimension:
    input_size = size(X)[1:end-1]

    # Buffer:
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    
    return ConditionalSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer, dt)
end

"""
    energy(sampler::ConditionalSampler, model, x, y)

Energy function for `ConditionalSampler`.
"""
function energy(sampler::ConditionalSampler, model, x, y)
    return _energy(model, x, y)
end

"""
    mcmc_samples(
        sampler::ConditionalSampler,
        model, rule::Flux.Optimise.AbstractOptimiser,
        inp_samples::AbstractArray;
        niter::Int=100, y::Union{Nothing,Int}=nothing
    )

Sampling method for `ConditionalSampler`.
"""
function mcmc_samples(
    sampler::ConditionalSampler,
    model,
    rule::Flux.Optimise.AbstractOptimiser,
    inp_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing
)
    # Setup
    if isnothing(y)
        y = rand(sampler.𝒟y)
    end
    E(x) = energy(sampler, model, x, y)
    rule = deepcopy(rule)

    # Training:
    i = 1
    while i <= niter
        Δ = gradient(E, inp_samples)[1]
        Δ = apply!(rule, inp_samples, Δ)
        inp_samples -= Δ
        i += 1
    end

    return inp_samples
end

@doc raw"""
    UnonditionalSampler <: AbstractSampler

Generates unconditional samples: $x \sim p(x).$
"""
mutable struct UnconditionalSampler <: AbstractSampler
    𝒟x::Distribution
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
end

"""
    UnconditionalSampler(
        𝒟x::Distribution;
        input_size::Dims, batch_size::Int,
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `UnonditionalSampler`.
"""
function UnconditionalSampler(
    𝒟x::Distribution;
    input_size::Dims, batch_size::Int=1,
    max_len::Int=10000, prob_buffer::AbstractFloat=0.95
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return UnconditionalSampler(𝒟x, input_size, batch_size, buffer, max_len, prob_buffer)
end

"""
    energy(sampler::UnconditionalSampler, model, x, y)

Energy function for `UnconditionalSampler`.
"""
function energy(sampler::UnconditionalSampler, model, x, y)
    return _energy(model, x)
end

"""
    mcmc_samples(
        sampler::UnconditionalSampler,
        model, rule::Flux.Optimise.AbstractOptimiser,
        inp_samples::AbstractArray;
        niter::Int=100
    )

Sampling method for `UnconditionalSampler`.
"""
function mcmc_samples(
    sampler::UnconditionalSampler,
    model,
    rule::Flux.Optimise.AbstractOptimiser,
    inp_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing
)

    # Setup:
    E(x) = energy(sampler, model, x, nothing)

    # Training:
    i = 1
    while i <= niter
        Δ = gradient(E, inp_samples)[1]
        Δ = apply!(rule, inp_samples, Δ)
        inp_samples -= Δ
        i += 1
    end

    return inp_samples

end

@doc raw"""
    JointSampler <: AbstractSampler

Generates unconditional samples by drawing directly from joint distribution: $x \sim p(x, y).$
"""
mutable struct JointSampler <: AbstractSampler
    𝒟x::Distribution
    𝒟y::Distribution
    input_size::Dims
    batch_size::Int
    buffer::AbstractArray
    max_len::Int
    prob_buffer::AbstractFloat
end

"""
    JointSampler(
        𝒟x::Distribution, 𝒟y::Distribution, input_size::Dims, batch_size::Int;
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `JointSampler`.
"""
function JointSampler(
    𝒟x::Distribution, 𝒟y::Distribution;
    input_size::Dims, batch_size::Int=1,
    max_len::Int=10000, prob_buffer::AbstractFloat=0.95
)
    @assert batch_size <= max_len "batch_size must be <= max_len"
    buffer = Float32.(rand(𝒟x, input_size..., batch_size))
    return JointSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer)
end

"""
    energy(sampler::JointSampler, model, x, y)

Energy function for `JointSampler`.
"""
function energy(sampler::JointSampler, model, x, y)
    return _energy(model, x, y)
end

"""
    mcmc_samples(
        sampler::JointSampler,
        model, rule::Flux.Optimise.AbstractOptimiser,
        inp_samples::AbstractArray;
        niter::Int=100
    )

Sampling method for `JointSampler`.
"""
function mcmc_samples(
    sampler::JointSampler,
    model,
    rule::Flux.Optimise.AbstractOptimiser,
    inp_samples::AbstractArray;
    niter::Int,
    y::Union{Nothing,Int}=nothing
)

    # Setup:
    E(x, y) = energy(sampler, model, x, y)

    # Training:
    i = 1
    while i <= niter
        y = rand(sampler.𝒟y)
        Δ = gradient(E, inp_samples, y)[1]
        Δ = apply!(rule, inp_samples, Δ)
        inp_samples -= Δ
        i += 1
    end

    return inp_samples

end

end