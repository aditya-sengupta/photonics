using Flux, ProgressMeter
using Statistics
using NPZ
using Plots
using Plots: heatmap
import Chain: @chain
using ZernikePolynomials
using Base.GC: gc
using CUDA
using JLD2

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin

X_all, xmin, xmax = @chain npzread("data/sim_trainsets/sim_trainset_amplitudes_240502_1947.npy") transpose Matrix zero_one_ify gpu
y_all, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_240502_1947.npy") abs2.(_) transpose Matrix zero_one_ify gpu
# X_all, y_all = Matrix{Float32}(X_all), Matrix{Float32}(y_all)

cutoff = Int(round(0.8 * size(X_all, 2)))
X_train, X_test = X_all[:,1:cutoff], X_all[:,cutoff+1:end]
y_train, y_test = y_all[:,1:cutoff], y_all[:,cutoff+1:end]

nzern = size(X_all, 1)
ws1, ws2 = 2000, 100
# ws1, ws2 = 200 * (nzern + 1), 10 * (nzern + 1)

model = Chain(
    Dense(size(y_all, 1) => ws1, relu),
    Dense(ws1 => ws2, relu),
    Flux.Dropout(0.2),
    Dense(ws2 => size(X_all, 1))
) |> gpu

optim = Flux.setup(Adam(1e-3), model)
loss(model, y, X) = Flux.mse(model(y), X)
loader = Flux.DataLoader((y_train, X_train), batchsize=32, shuffle=true) |> gpu

p = plot(xlabel="Epoch", ylabel="Loss", yscale=:log10)

begin
    k = 1
    losses = []
    last_loss = loss(model, y_train, X_train)
    @showprogress for epoch in 1:200
        for (y, X) in loader
            Flux.train!(
                loss,
                model,
                [(y, X)],
                optim
            )
        end
        l = loss(model, y_train, X_train)
        if epoch % k == 0
            plot!(p, [epoch - k, epoch], [last_loss, l], label=nothing, color=1)
            last_loss = l
            display(p)
        end
        push!(losses, l)
        if epoch % 10 == 0
            gc()
        end
    end
end
loss(model, y_train, X_train)
loss(model, y_test, X_test)

error_per_zern = mean(abs, X_test .- model(y_test), dims=2) |> cpu
plot(error_per_zern, xlabel="Zernike mode", ylabel="Fractional error", label=nothing)

model_state = Flux.state(cpu(model))
jldsave("data/pl_nn/pl_nn_$(round(xmax, digits=2))_$(size(X_all, 1)).jld2"; model_state)

# for the phase screen test stuff, look on the git history