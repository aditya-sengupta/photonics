using Flux, ProgressMeter
using Statistics
using NPZ
using Plots
using Plots: heatmap
import Chain: @chain
using ZernikePolynomials

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin

X_all, xmin, xmax = @chain npzread("data/pl_230808/inputzs_230808_1346.npy") transpose Matrix zero_one_ify
y_all, ymin, ymax = @chain npzread("data/pl_230808/pls_230808_1346.npy") transpose Matrix zero_one_ify

X_all, y_all = Matrix{Float32}(X_all), Matrix{Float32}(y_all)

cutoff = Int(round(0.8 * size(X_all, 2)))
X_train, X_test = X_all[:,1:cutoff], X_all[:,cutoff+1:end]
y_train, y_test = y_all[:,1:cutoff], y_all[:,cutoff+1:end]

model = Chain(
    Dense(size(y_all, 1) => 2000, relu),
    Dense(2000 => 1000, relu),
    Dense(1000 => size(X_all, 1))
)

optim = Flux.setup(Adam(1e-4), model)
λ = 1e-3
loss(model, y, X) = Flux.mse(model(y), X) + λ * sum(sum(relu, -layer.weight) for layer in model.layers)
loader = Flux.DataLoader((y_train, X_train), batchsize=32)

p = plot(xlabel="Epoch", ylabel="Loss")

begin
    k = 1
    losses = []
    last_loss = loss(model, y_train, X_train)
    @showprogress for epoch in 1:40
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
    end
end
loss(model, y_train, X_train)
loss(model, y_test, X_test)

error_per_zern = mean(abs, X_test .- model(y_test), dims=2)
plot(error_per_zern, xlabel="Zernike mode", ylabel="Percent error", label=nothing)

im_show(x; kwargs...) = heatmap(x, aspect_ratio=1, showaxis=false, grid=false, xlim=(0, size(x, 1) + 2); kwargs...)

zv = collect(1:size(X_all, 1))

function get_phase_screens(i, N=256)
    xp = rescale.(X_test[:,i], xmin, xmax)
    yp = y_test[:,i]
    rxp = rescale.(Vector{Float64}(model(yp)), xmin, xmax)
    z_init = evaluateZernike(N, zv, xp);
    z_recon = evaluateZernike(N, zv, rxp);
    z_resid = evaluateZernike(N, zv, xp - rxp);
    return z_init, z_recon, z_resid
end

begin
    rms_resids = []
    @showprogress for i in 1:size(X_test, 2)
        z_init, z_recon, z_resid = get_phase_screens(i, 32)
        push!(rms_resids, mean(abs2, z_resid) / mean(abs2, z_init))
    end
end

begin
    for i in [argmin(rms_resids), argmax(rms_resids)]
        z_init, z_recon, z_resid = get_phase_screens(i)
        clims = (
            min(minimum(z_init), minimum(z_recon), minimum(z_resid)),
            max(maximum(z_init), maximum(z_recon), maximum(z_resid))
        )
        p1 = im_show(z_init, title="Initial phase screen", titlefontsize=10, clims=clims)
        p2 = im_show(z_recon, title="NN reconstruction", titlefontsize=10, clims=clims)
        p3 = im_show(z_resid, title="NN residual, rms error = $(round(mean(abs2, z_resid), digits=4)) rad", titlefontsize=10, clims=clims)
        p = plot(p1, p2, p3)
        Plots.savefig("figures/initial_nn_test_$i.pdf")
    end
end