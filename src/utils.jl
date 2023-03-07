using StatsBase

@doc raw"""
    energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
energy(f, x; agg=mean) = agg(map(x -> -logsumexp(f(x)), eachslice(x, dims=ndims(x))))

@doc raw"""
    energy(f, x, y::Int; agg=mean)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function energy(f, x, y::Int; agg=mean)
    agg(map(x -> -f(x)[y], eachslice(x, dims=ndims(x))))
end



