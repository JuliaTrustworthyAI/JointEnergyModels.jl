setup_docs = quote

    using Pkg
    Pkg.activate("docs")


    using Distributions
    using Flux
    using Flux.Data: DataLoader
    using Flux: onehotbatch, onecold, @epochs, @functor, logsumexp
    using Flux.Losses: logitcrossentropy
    using Images
    using JointEnergyModels
    using MLDatasets
    using MLJBase
    using Plots
    using Plots.PlotMeasures
    using Random
    using Statistics

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "docs/src/www"
    include("docs/src/utils.jl")
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

end;
