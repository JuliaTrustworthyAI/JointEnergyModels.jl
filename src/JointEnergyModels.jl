module JointEnergyModels

using Flux
abstract type AbstractSamplingRule <: Flux.Optimise.AbstractOptimiser end
abstract type AbstractSampler end

include("utils.jl")
export energy

include("model.jl")
export JointEnergyModel
export class_loss, gen_loss, loss

include("Optimisers.jl")
using .Optimisers
export SGLD, ImproperSGLD

include("Samplers.jl")
using .Samplers
export ConditionalSampler, UnconditionalSampler

include("training.jl")

end
