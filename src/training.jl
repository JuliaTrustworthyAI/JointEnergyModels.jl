using Flux: onecold
using Flux.Data: DataLoader
using StatsBase


function accuracy(jem::JointEnergyModel, x, y; agg=mean)
    ŷ = jem(x)
    agg(onecold(ŷ) .== onecold(y))
end

function evaluation(jem::JointEnergyModel, val_set::DataLoader)
    ℓ = 0.0
    ℓ_clf = 0.0
    ℓ_gen = 0.0
    ℓ_reg = 0.0
    acc = 0.0
    num = 0
    for (x, y) in val_set
        ℓ_clf += sum(JointEnergyModels.class_loss(jem, x, y))
        ℓ_gen += sum(JointEnergyModels.gen_loss(jem, x, y))
        ℓ_reg += sum(JointEnergyModels.reg_loss(jem, x, y))
        ℓ += JointEnergyModels.loss(jem, x, y)
        acc += accuracy(jem, x, y)
        num += size(x)[end]
    end
    return ℓ / num, ℓ_clf / num, ℓ_gen / num, ℓ_reg / num, acc / length(val_set)
end

function train_model(
    jem::JointEnergyModel, train_set, opt_state;
    num_epochs::Int=100, 
    val_set::Union{Nothing,DataLoader}=nothing, 
    max_patience::Int=10,
    verbosity::Int=num_epochs,
    use_class_loss::Bool=true, 
    use_gen_loss::Bool=true, 
    use_reg_loss::Bool=true,
    α::Float64=0.1,
)
    training_log = []
    not_finite_counter = 0

    for epoch in 1:num_epochs
        training_losses = Float32[]

        # Training:
        for (i, data) in enumerate(train_set)

            # Forward pass:
            x, y = data
            val, grads = Flux.withgradient(jem) do m
                JointEnergyModels.loss(
                    m, x, y; 
                    use_class_loss=use_class_loss, 
                    use_gen_loss=use_gen_loss, 
                    use_reg_loss=use_reg_loss,
                    α=α,
                )
            end

            # Save the loss from the forward pass. (Done outside of gradient.)
            push!(training_losses, val)

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                continue
            end

            Flux.update!(opt_state, jem, grads[1])
        end

        # Detect if loss has been Inf or NaN for 10 consecutive batches and break.
        if any(!isfinite, training_losses)
            not_finite_counter += 1
            if not_finite_counter == 10
                @warn "Loss not Inf or NaN for 10 epochs. Stopping training."
                break
            end
        end

        # Evaluation:
        if !isnothing(val_set)
            ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc = evaluation(jem, val_set)
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc, training_losses))
        else
            ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc = evaluation(jem, train_set)
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc, training_losses))
        end

        # Verbosity:
        if (verbosity > 0) && (epoch % round(num_epochs / verbosity) == 0)
            if isnothing(val_set)
                @info "Traning losses/accuracy in epoch $epoch:"
                println("Classification: $ℓ_clf")
                println("Generative: $ℓ_gen")
                println("Regularisation: $ℓ_reg")
                println("Total: $ℓ")
                println("Accuracy: $acc")
            else
                @info "Validation losses/accuracy in epoch $epoch:"
                println("Classification: $ℓ_clf")
                println("Generative: $ℓ_gen")
                println("Regularisation: $ℓ_reg")
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