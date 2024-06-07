using Flux
using StatsBase

get_logits(f::Flux.Chain, x) = f[end] isa Function ? f[1:end-1](x) : f(x)

@doc raw"""
    energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
function _energy(f, x; agg = mean)
    if f isa Flux.Chain
        ŷ = get_logits(f, x)
    else
        ŷ = f(x)
    end
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(y -> -logsumexp(y), eachslice(ŷ, dims = ndims(ŷ))))
        return E
    else
        return -logsumexp(ŷ)
    end
end

@doc raw"""
    energy(f, x, y::Int; agg=mean)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function _energy(f, x, y::Int; agg = mean)
    if f isa Flux.Chain
        ŷ = get_logits(f, x)
    else
        ŷ = f(x)
    end
    _E(y, idx) = length(y) > 1 ? -y[idx] : (idx == 2 ? -y[1] : -(1.0 - y[1]))
    if ndims(ŷ) > 1
        E = 0.0
        E = agg(map(_y -> _E(_y, y), eachslice(ŷ, dims = ndims(ŷ))))
        return E
    else
        return _E(_y, y)
    end
end
