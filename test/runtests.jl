using Distributions
using Flux
using JointEnergyModels
using MLJBase
using MLJFlux
using Test

include("utils.jl")

@testset "JointEnergyModels.jl" begin

    include("aqua.jl")

    @testset "mlj_flux.jl" begin
        include("mlj_flux.jl")
    end

end
