𝒟x = Normal()
𝒟y = Categorical(ones(2) ./ 2)
sampler = ConditionalSampler(𝒟x, 𝒟y, input_size=size(Xmat)[1:end-1], batch_size=batch_size)

@testset "ConditionalSampler" begin
    @testset "constructor" begin
        @testset "default" begin
            @test sampler isa ConditionalSampler
            @test sampler isa AbstractSampler
        end
    end
end