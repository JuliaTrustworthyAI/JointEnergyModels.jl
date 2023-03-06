using StatsBase

@doc raw"""
    energy(f, x; dims=Union{Nothing, Int}=nothing)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
function energy(f, x; dims::Union{Nothing,Int} = nothing)
    if isnothing(dims)
        E = -logsumexp(f(x))
    else
        E = map(x -> -logsumexp(f(x)), eachslice(x, dims=dims))
    end
    return E
end

@doc raw"""
    energy(f, x, y::Int; dims=Union{Nothing,Int} = nothing)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function energy(f, x, y::Int; dims::Union{Nothing,Int} = nothing)
    if isnothing(dims)
        E = -f(x)[y]
    else
        E = map(x -> -f(x)[y], eachslice(x, dims=dims))
    end
    return E
end

# function StatsBase.sample(jem::JointEnergyModel, sampler::AbstractSampler, dims::Dims; unconditional=false)

# end