module JointEnergyModels

using EnergySamplers
using Flux
using TaijaBase

using Reexport
@reexport import EnergySamplers: ConditionalSampler, UnconditionalSampler, JointSampler

include("model.jl")
export JointEnergyModel
export class_loss, gen_loss, loss
export generate_samples, generate_conditional_samples

include("training.jl")
export train_model

include("mlj_flux.jl")
export JointEnergyClassifier

include("samplers.jl")

end
