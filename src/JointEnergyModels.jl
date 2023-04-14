module JointEnergyModels

using Flux
abstract type AbstractSamplingRule <: Flux.Optimise.AbstractOptimiser end
abstract type AbstractSampler end

include("utils.jl")
export _energy

include("Samplers.jl")
using .Samplers
export ConditionalSampler, UnconditionalSampler
export energy

include("model.jl")
export JointEnergyModel
export class_loss, gen_loss, loss
export generate_samples, generate_conditional_samples

include("Optimisers.jl")
using .Optimisers
export SGLD, ImproperSGLD

include("training.jl")
export train_model

end
