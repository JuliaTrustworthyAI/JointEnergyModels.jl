setup_docs = quote

    using Pkg
    Pkg.activate("docs")


    using Distributions
    using Flux
    using Flux.Data: DataLoader
    using Flux: onehotbatch, onecold, @functor, logsumexp
    using Flux.Losses: logitcrossentropy
    using Images
    using JointEnergyModels
    using MLDatasets
    using MLJBase
    using MLJFlux
    using MLUtils
    using Plots
    using Plots.PlotMeasures
    using Random
    using Statistics
    using TaijaBase
    using EnergySamplers: ConditionalSampler

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "$(pwd())/docs/src/www"
    include("$(pwd())/docs/src/utils/utils.jl")
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

end;
