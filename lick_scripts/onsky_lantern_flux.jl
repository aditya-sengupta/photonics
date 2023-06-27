using Flux, ProgressMeter
using Statistics
using StatsBase: sample
using NPZ
using Plots
import Chain: @chain
using ZernikePolynomials
using PhotonicLantern
using InvertedIndices

X_all, xmin, xmax = @chain npzread("data/pl_230602/inputzs_onsky1.npy")[:,1:18] transpose convert(Matrix{Float32}, _) zero_one_ify
y_all, ymin, ymax = @chain npzread("data/pl_230602/outputls_onsky1.npy") transpose convert(Matrix{Float32}, _) zero_one_ify

n = size(X_all, 2)
n_split = Int(round(0.8 * n))
train_mask = sample(1:n,n_split,replace=false)
test_mask = Not(train_mask)

X_train, X_test = X_all[:,train_mask], X_all[:,test_mask]
y_train, y_test = y_all[:,train_mask], y_all[:,test_mask]

model = Chain(
    Dense(size(y_all, 1) => 1000, relu),
    Dense(1000 => 500, relu),
    Dense(500 => 500, relu),
    Dense(500 => size(X_all, 1))
)

optim = Flux.setup(Adam(1e-8), model)
λ = 0.0
loss(model, y, X) = Flux.mse(model(y), X) #+ λ * sum(sum(relu, -layer.weight) for layer in model.layers)
loader = Flux.DataLoader((y_train, X_train), batchsize=32)

p = plot(xlabel="Epoch", ylabel="Loss", yscale=:log10)
start_epoch = 0
begin
    kc = 1
    n_epochs = 20
    losses = []
    last_loss = loss(model, y_train, X_train)
    @showprogress for epoch in (start_epoch+1):(start_epoch + n_epochs)
        for (y, X) in loader
            Flux.train!(
                loss,
                model,
                [(y, X)],
                optim
            )
        end
        l = loss(model, y_train, X_train)
        if epoch % kc == 0
            plot!(p, [epoch - kc, epoch], [last_loss, l], label=nothing, color=1)
            last_loss = l
            display(p)
        end
        push!(losses, l)
    end
    start_epoch += n_epochs
end

loss(model, y_train, X_train)
loss(model, y_test, X_test)

loss(x -> x * 0 .+ mean(y_train, dims=2), y_test, X_test)

error_per_zern = mean(abs, X_test .- model(y_test), dims=2)
plot(error_per_zern * 100, xlabel="Zernike mode", ylabel="Percent error", label=nothing)

zv = collect(1:size(X_all, 1))

function get_phase_screens(i, zsize=256)
    xp = rescale(X_test[:,i], xmin, xmax)
    yp = y_test[:,i]
    rxp = rescale(Vector{Float64}(model(yp)), xmin, xmax)
    z_init = phasewrap.(evaluateZernike(zsize, zv, Vector{Float64}(xp)));
    z_recon = phasewrap.(evaluateZernike(zsize, zv, rxp));
    z_resid = phasewrap.(evaluateZernike(zsize, zv, xp - rxp));
    return z_init, z_recon, z_resid
end

function plot_phase_screens(z_init, z_recon, z_resid)
    clims = (
        min(minimum(z_init), minimum(z_recon), minimum(z_resid)),
        max(maximum(z_init), maximum(z_recon), maximum(z_resid))
    )
    p1 = im_show(z_init, title="Initial phase screen", titlefontsize=10, clims=clims)
    p2 = im_show(z_recon, title="NN reconstruction", titlefontsize=10, clims=clims)
    p3 = im_show(z_resid, title="Residual, rms error = $(round(sqrt(mean(abs2, z_resid)), digits=4)) rad", titlefontsize=10, clims=clims)
    p = plot(p1, p2, p3)
end

plot_phase_screens(i) = plot_phase_screens(get_phase_screens(i)...)

begin
    a = Animation()
    @showprogress for i = 1:10:(n-n_split)
        frame(a, plot_phase_screens(i))
    end
    gif(a, "figures/onsky_nn.gif", fps=2)
end

begin
    rms_resids = []
    @showprogress for i in 1:(n-n_split)
        _, _, z_resid = get_phase_screens(i, 32)
        push!(rms_resids, mean(abs2, z_resid))
    end
end

begin
    for i in [argmax(rms_resids), argmin(rms_resids)]
        p = plot_phase_screens(i)
        Plots.savefig("figures/onsky_nn_test_$i.pdf")
    end
    p
end