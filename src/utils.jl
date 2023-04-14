using StatsBase

@doc raw"""
    _energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
function _energy(f, x::AbstractArray; agg=mean)
    ŷ = f(x)
    E = agg(map(y -> -logsumexp(y), eachslice(ŷ, dims=ndims(ŷ))))
    return E
end

@doc raw"""
    _energy(f, x, y::Int; agg=mean)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function _energy(f, x::AbstractArray, y::Int; agg=mean)
    ŷ = f(x)
    E = 0.0
    E = agg(map(_y -> -_y[y], eachslice(ŷ, dims=ndims(ŷ))))
    return E
end



