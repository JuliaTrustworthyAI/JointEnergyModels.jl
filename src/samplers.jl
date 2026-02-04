using CategoricalArrays
using Distributions

"""
    ConditionalSampler(
        X::AbstractArray, y::AbstractArray;
        batch_size::Int,
        max_len::Int=10000, prob_buffer::AbstractFloat=0.95
    )

Outer constructor for `ConditionalSampler`.
"""
function EnergySamplers.ConditionalSampler(
    X::Union{Tables.MatrixTable,AbstractMatrix},
    y::Union{CategoricalArray,AbstractMatrix};
    batch_size::Int = 1,
    max_len::Int = 10000,
    prob_buffer::AbstractFloat = 0.95,
)
    @assert batch_size <= max_len "batch_size must be <= max_len"

    # Preprocess data:
    X = X isa Tables.MatrixTable ? MLJFlux.reformat(X) : X
    y = y isa CategoricalArray ? MLJFlux.reformat(y) : y

    # Prior distributions:
    𝒟x = Uniform(extrema(X)...)
    n_classes = size(y, 1)
    𝒟y = Categorical(ones(n_classes) ./ n_classes)          # TODO: make more general

    # Input dimension:
    input_size = size(X)[1:(end-1)]

    # Buffer:
    buffer = Float32.(rand(𝒟x, input_size..., maximum([1000, batch_size])))

    return ConditionalSampler(𝒟x, 𝒟y, input_size, batch_size, buffer, max_len, prob_buffer)
end
