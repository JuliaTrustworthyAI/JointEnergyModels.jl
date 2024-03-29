using Distributions
using Flux
using JointEnergyModels
using MLJBase
using MLJFlux
using Test

import CompatHelperLocal as CHL
CHL.@check()

include("utils.jl")

@testset "JointEnergyModels.jl" begin
    
    @testset "samplers.jl" begin
        include("samplers.jl")
    end

    @testset "mlj_flux.jl" begin
        include("mlj_flux.jl")
    end

end
