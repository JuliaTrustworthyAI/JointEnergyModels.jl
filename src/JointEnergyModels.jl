module JointEnergyModels

using Flux

abstract type AbstractSampler <: Flux.Optimise.AbstractOptimiser end

include("Samplers.jl")

end
