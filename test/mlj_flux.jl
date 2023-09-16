@testset "JointEnergyClassifier" begin
    @testset "CPU" begin
        clf = JointEnergyClassifier(
            sampler;
            builder=MLJFlux.MLP(hidden=(32, 32, 32,), σ=Flux.relu),
            batch_size=batch_size,
            finaliser=x -> x,
            loss=Flux.Losses.logitcrossentropy,
            jem_training_params=(α=[1.0, 1.0, 0.1], verbosity=5,)
        )
        mach = machine(clf, X, y)

        @testset "constructor" begin
            @testset "default" begin
                @test clf isa MLJFlux.MLJFluxModel
                @test clf isa JointEnergyClassifier
            end
        end

        @testset "fitting" begin
            fit!(mach)
        end
    end

    @testset "GPU" begin
        @info "CUDA is functional: $(CUDA.functional())."
        clf = JointEnergyClassifier(
            sampler;
            builder=MLJFlux.MLP(hidden=(32, 32, 32,), σ=Flux.relu),
            batch_size=batch_size,
            finaliser=x -> x,
            loss=Flux.Losses.logitcrossentropy,
            jem_training_params=(α=[1.0, 1.0, 0.1], verbosity=5,),
            acceleration = CUDALibs()
        )
        mach = machine(clf, X, y)
        
        @testset "constructor" begin
            @testset "default" begin
                @test clf isa MLJFlux.MLJFluxModel
                @test clf isa JointEnergyClassifier
            end
        end

        @testset "fitting" begin
            fit!(mach)
        end
    end
end
