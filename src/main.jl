using StatsBase

@doc raw"""
    energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
energy(f, x) = -logsumexp(f(x))

@doc raw"""
    energy(f, x, y::Int)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
energy(f, x, y::Int) = -f(x)[y]

