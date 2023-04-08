using Flux: onecold
using Flux.Data: DataLoader


function accuracy(jem::JointEnergyModel, x, y; agg=mean)
    ŷ = jem(x)
    agg(onecold(ŷ) .== onecold(y))
end

function evaluation(jem::JointEnergyModel, val_set::DataLoader)
    ℓ = 0.0
    ℓ_clf = 0.0
    ℓ_gen = 0.0
    acc = 0.0
    num = 0
    for (x, y) in val_set
        ℓ_clf += sum(JointEnergyModels.class_loss(jem, x, y))
        ℓ_gen += sum(JointEnergyModels.gen_loss(jem, x))
        ℓ += JointEnergyModels.loss(jem, x, y)
        acc += accuracy(jem, x, y)
        num += size(x)[end]
    end
    return ℓ / num, ℓ_clf / num, ℓ_gen / num, acc / length(val_set)
end

function training(
    jem::JointEnergyModel, train_set, opt_state;
    num_epochs::Int=100, val_set::Union{Nothing,DataLoader}=nothing, max_patience::Int=20,
    verbosity::Int=num_epochs
)
    training_log = []
    for epoch in 1:num_epochs
        training_losses = Float32[]

        # Training:
        for (i, data) in enumerate(train_set)

            # Forward pass:
            x, y = data
            val, grads = Flux.withgradient(jem) do m
                JointEnergyModels.loss(m, x, y)
            end

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(training_losses, val)

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_state, jem, grads[1])
        end

        # Evluation:
        if !isnothing(val_set)
            ℓ, ℓ_clf, ℓ_gen, acc = evaluation(jem, val_set)
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, acc, training_losses))
        else
            ℓ, ℓ_clf, ℓ_gen, acc = evaluation(jem, train_set)
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, acc, training_losses))
        end

        # Verbosity:
        if (verbosity > 0) && (epoch % round(num_epochs / verbosity) == 0)
            if isnothing(val_set)
                @info "Traning losses/accuracy in epoch $epoch:"
                println("Classification: $ℓ_clf")
                println("Generative: $ℓ_gen")
                println("Total: $ℓ")
                println("Accuracy: $acc")
            else
                @info "Validation losses/accuracy in epoch $epoch:"
                println("Classification: $ℓ_clf")
                println("Generative: $ℓ_gen")
                println("Total: $ℓ")
                println("Accuracy: $acc")
            end
        end

        # Early Stopping:
        _loss() = ℓ
        es = Flux.early_stopping(_loss, max_patience)
        es() && break

    end

    return training_log
end