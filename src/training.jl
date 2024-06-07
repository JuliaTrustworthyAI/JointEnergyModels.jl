using Flux: onecold
using Flux: DataLoader
using Flux.Losses: logitcrossentropy
using ProgressMeter

function accuracy(jem::JointEnergyModel, x, y; agg = mean)
    ŷ = jem(x)
    agg(onecold(ŷ) .== onecold(y))
end

function evaluation(
    jem::JointEnergyModel,
    val_set::Union{DataLoader,Base.Iterators.Zip};
    class_loss_fun::Function = logitcrossentropy,
    use_class_loss::Bool = true,
    use_gen_loss::Bool = true,
    use_reg_loss::Bool = true,
    α = [1.0, 1.0, 1e-1],
)
    ℓ = 0.0
    ℓ_clf = 0.0
    ℓ_gen = 0.0
    ℓ_reg = 0.0
    acc = 0.0
    num = 0
    for (x, y) in val_set
        ℓ_clf += sum(JointEnergyModels.class_loss(jem, x, y; loss_fun = class_loss_fun))
        ℓ_gen += sum(JointEnergyModels.gen_loss(jem, x, y))
        ℓ_reg += sum(JointEnergyModels.reg_loss(jem, x, y))
        ℓ += JointEnergyModels.loss(
            jem,
            x,
            y;
            use_class_loss = use_class_loss,
            use_gen_loss = use_gen_loss,
            use_reg_loss = use_reg_loss,
            class_loss_fun = class_loss_fun,
            α = α,
        )
        acc += accuracy(jem, x, y)
        num += size(x)[end]
    end
    return ℓ / num, ℓ_clf / num, ℓ_gen / num, ℓ_reg / num, acc / length(val_set)
end

function train_model(
    jem::JointEnergyModel,
    train_set,
    opt_state;
    num_epochs::Int = 100,
    val_set::Union{Nothing,DataLoader,Base.Iterators.Zip} = nothing,
    max_patience::Int = 10,
    verbosity::Int = num_epochs,
    use_class_loss::Bool = true,
    use_gen_loss::Bool = true,
    use_reg_loss::Bool = true,
    α = [1.0, 1.0, 1e-1],
    class_loss_fun::Function = logitcrossentropy,
    progress_meter::Union{Nothing,ProgressMeter.Progress} = nothing,
)
    training_log = []
    not_finite_counter = 0
    if isnothing(progress_meter)
        progress_meter = Progress(
            num_epochs,
            dt = 0,
            desc = "Optimising neural net:",
            barglyphs = BarGlyphs("[=> ]"),
            barlen = 25,
            color = :green,
        )
        verbosity == 0 || next!(progress_meter)
    end

    for epoch = 1:num_epochs
        training_losses = Float32[]

        # Training:
        for (i, data) in enumerate(train_set)

            # Forward pass:
            x, y = data
            val, grads = Flux.withgradient(jem) do m
                JointEnergyModels.loss(
                    m,
                    x,
                    y;
                    use_class_loss = use_class_loss,
                    use_gen_loss = use_gen_loss,
                    use_reg_loss = use_reg_loss,
                    α = α,
                    class_loss_fun = class_loss_fun,
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
            ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc = evaluation(
                jem,
                val_set;
                use_class_loss = use_class_loss,
                use_gen_loss = use_gen_loss,
                use_reg_loss = use_reg_loss,
                class_loss_fun = class_loss_fun,
                α = α,
            )
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc, training_losses))
        else
            ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc = evaluation(
                jem,
                train_set;
                use_class_loss = use_class_loss,
                use_gen_loss = use_gen_loss,
                use_reg_loss = use_reg_loss,
                class_loss_fun = class_loss_fun,
                α = α,
            )
            push!(training_log, (; ℓ, ℓ_clf, ℓ_gen, ℓ_reg, acc, training_losses))
        end

        # Verbosity:
        verbosity == 0 || next!(progress_meter)
        if (verbosity > 0) && (epoch % round(num_epochs / verbosity) == 0)
            println("")
            if isnothing(val_set)
                @info "Training losses/accuracy in epoch $epoch:"
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
