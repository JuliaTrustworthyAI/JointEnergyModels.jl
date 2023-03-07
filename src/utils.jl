using StatsBase

@doc raw"""
    energy(f, x)

Computes the energy for unconditional samples $x \sim p_{\theta}(x)$: $E(x)=-\text{LogSumExp}_y f_{\theta}(x)[y]$.
"""
function energy(f, x; dims::Union{Nothing,Int}=2) 
    ŷ = f(x)
    if isnothing(dims)
        return -logsumexp(ŷ)
    else
        return map(y -> -logsumexp(y), eachslice(ŷ; dims=dims))
    end
end

@doc raw"""
    energy(f, x, y::Int)

Computes the energy for conditional samples $x \sim p_{\theta}(x|y)$: $E(x)=- f_{\theta}(x)[y]$.
"""
function energy(f, x, y::Int) 
    -f(x)'Flux.onehot(y,1:size(f(x),1))
end



