𝒟x = Normal()
𝒟y = Categorical(ones(2) ./ 2)
sampler = ConditionalSampler(𝒟x, 𝒟y, input_size=size(Xmat)[1:end-1], batch_size=batch_size)

clf = JointEnergyClassifier(
    sampler;
    builder=MLJFlux.MLP(hidden=(32, 32, 32,), σ=Flux.relu),
    batch_size=batch_size,
    finaliser=x -> x,
    loss=Flux.Losses.logitcrossentropy,
    jem_training_params=(α=[1.0, 1.0, 0.1], verbosity=5,)
)
mach = machine(clf, X, y)

@testset "Conditional from data" begin
    ConditionalSampler(X, y)
    @test true
end

@testset "JointEnergyClassifier" begin
    @testset "constructor" begin
        @testset "default" begin
            @test clf isa MLJFlux.MLJFluxModel
            @test clf isa JointEnergyClassifier
        end
    end

    @testset "fit" begin
        @testset "default" begin
            fit!(mach)
        end
    end
end
