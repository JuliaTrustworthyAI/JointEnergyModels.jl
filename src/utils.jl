using StatsBase

@doc raw"""
    energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
function _energy(f, x; agg=mean)
    ŷ = f(x)
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(y -> -logsumexp(y), eachslice(ŷ, dims=ndims(ŷ))))
        return E
    else
        return -logsumexp(ŷ)
    end
end

@doc raw"""
    energy(f, x, y::Int; agg=mean)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function _energy(f, x, y::Int; agg=mean)
    ŷ = f(x)
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(_y -> -_y[y], eachslice(ŷ, dims=ndims(ŷ))))
        return E
    else
        return -ŷ[y]
    end
end



