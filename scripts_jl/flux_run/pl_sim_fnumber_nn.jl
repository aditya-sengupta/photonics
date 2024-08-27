using Pkg
Pkg.activate(".")
using Flux, ProgressMeter
using Statistics
using NPZ
using Plots
using Plots: heatmap
import Chain: @chain
using ZernikePolynomials
using Base.GC: gc
using CUDA
using CUDA: @allowscalar
using JLD2

function zero_one_ify(x, xmin=nothing, xmax=nothing)
    if isnothing(xmin) && isnothing(xmax)
        xmin, xmax = minimum(x), maximum(x)
    end
    return (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
end

rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin

for f in 3:11
    X_train, xmin, xmax = @chain npzread("data/sim_trainsets/fnumber/sim_trainset_amplitudes_fnumber_$(f)_date_240822.npy") transpose Matrix zero_one_ify gpu
    y_train, ymin, ymax = @chain npzread("data/sim_trainsets/fnumber/sim_trainset_lanterns_fnumber_$(f)_date_240822.npy") abs2.(_) transpose Matrix zero_one_ify gpu
    X_test = @chain npzread("data/sim_trainsets/fnumber/sim_testset_amplitudes_fnumber_$(f)_date_240822.npy") transpose Matrix zero_one_ify(_, xmin, xmax) (_[1]) gpu
    y_test = @chain npzread("data/sim_trainsets/fnumber/sim_testset_lanterns_fnumber_$(f)_date_240822.npy") abs2.(_) transpose Matrix zero_one_ify(_, ymin, ymax) (_[1]) gpu

    nzern = size(X_train, 1)
    ws1 = 200
    ws2_start = 20
    losses_f = Float64[]
    model = Chain(
        Dense(size(y_train, 1) => ws1, relu), # mapping layer
        Dense(ws1 => ws2_start), # bottleneck layer
        Flux.Scale(trues(ws2_start)), # cutoff for bottleneck layer
        Dense(ws2_start => ws1, relu), # demapping layer
        Dense(ws1 => size(X_train, 1)) # output layer
    ) |> gpu
    for ws2 in ws2_start:-1:0
        if ws2 < ws2_start
            @allowscalar model[3].scale[ws2+1] = 0
        end
        optim = Flux.setup(Adam(1e-3), model)
        loss(model, y, X) = Flux.mse(model(y), X)
        loader = Flux.DataLoader((y_train, X_train), batchsize=32, shuffle=true) |> gpu

        losses = []
        k = 20
        dloss = 1.0
        niter = 0
        prog = ProgressThresh(1e-3; desc="Iterating until loss is stable:")
        while dloss > 1e-3
            for (y, X) in loader
                Flux.train!(
                    loss,
                    model,
                    [(y, X)],
                    optim
                )
            end
            l = loss(model, y_train, X_train)
            push!(losses, l)
            niter += 1
            if niter > k
                dloss = (losses[niter-k] - l) / losses[niter-k]
            end
           update!(prog, Float64(dloss))
            if niter % 10 == 0
                gc()
            end
        end

        train_loss = minimum(losses)
        println("f/$f, ws2 = $ws2, loss = ", train_loss)
        push!(losses_f, train_loss)
        # model_state = Flux.state(cpu(model))
        # jldsave("data/pl_nn/pl_nn_fnumber_$f.jld2"; model_state)
    end

    npzwrite("data/pl_nn/test_losses_19_f_$(string(f)).npy", losses_f)
end
