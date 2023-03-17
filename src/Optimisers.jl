module Optimisers

using Flux
using ..JointEnergyModels
using ..JointEnergyModels: AbstractSamplingRule

export SGLD, ImproperSGLD

@doc raw"""
    SGLD(a::Real=1.0, b::Real=1.0, γ::Real=0.5)

Stochastic Gradient Langevin Dynamics ([SGLD](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)) optimizer.

# Examples
```julia
opt = SGLD()
opt = SGLD(2.0, 100.0, 0.9)
```
"""
struct SGLD <: AbstractSamplingRule
    a::Float64
    b::Float64
    gamma::Float64
    state::IdDict{Any,Any}
end
SGLD(a::Real=2.0, b::Real=1.0, γ::Real=0.9) = SGLD(a, b, γ, IdDict())

function Flux.Optimise.apply!(o::SGLD, x, Δ)
    a, b, γ = o.a, o.b, o.gamma

    t = get!(o.state, :t, Ref(1))
    εt = get!(o.state, t) do
        (zero(x),)
    end::Tuple{typeof(x)}

    εt = @.(a * (b + t)^-γ)
    ηt = εt .* Float32.(randn(size(Δ)))
    Δ = Float32.(@.(0.5 * εt * Δ + ηt))

    t[] += 1

    return Δ
end

@doc raw"""
    ImproperSGLD(a::Real=1.0, b::Real=1.0, γ::Real=0.5)

Improper [SGLD](https://openreview.net/pdf?id=Hkxzx0NtDB) optimizer.

# Examples
```julia
opt = ImproperSGLD()
opt = SGLD(2.0, 0.01)
```
"""
struct ImproperSGLD <: AbstractSamplingRule
    alpha::Float64
    sigma::Float64
end
ImproperSGLD(α::Real=2.0, σ::Real=0.01) = ImproperSGLD(α, σ)

function Flux.Optimise.apply!(o::ImproperSGLD, x, Δ)
    α, σ = o.alpha, o.sigma

    ηt = σ .* Float32.(randn(size(Δ)))
    Δ = Float32.(@.(0.5 * α * Δ + ηt))

    return Δ
end

end