using NPZ
using Flux
using JLD2
using ProgressMeter
using CairoMakie
using StatsBase: norm

zernikes = npzread("data/sim_trainsets/fnumber/sim_trainset_amplitudes_fnumber_5_date_240827.npy")
lanterns = Float32.(abs2.(npzread("data/sim_trainsets/fnumber/sim_trainset_lanterns_fnumber_5_date_240827.npy")))
model = Chain(
    Dense(19 => 4000, relu), # mapping layer
    Dense(4000 => 4000), # bottleneck layer
    Dense(4000 => 20, relu),
    Flux.Scale(trues(20))
)

ntrials = 10_000
wfs_distances = zeros(ntrials)
embed_dim_distances = Dict()
for k in [5, 9, 18]
    model_state = load("data/pl_nn/nlpca_identity_ws2_$k.jld2")["model_state"]
    model_state = (;zip((:layers,), [model_state.layers[1:4]])...)
    Flux.loadmodel!(model, model_state)
    ntrials = 10_000
    idxs_1 = rand(1:size(zernikes,1), ntrials)
    idxs_2 = rand(1:size(zernikes,1), ntrials)
    embed_dim_distances[k] = zeros(ntrials)
    @showprogress for (i, (idx1, idx2)) in enumerate(zip(idxs_1, idxs_2))
        # compute the wavefront sensor (WFS) distance
        if wfs_distances[i] == 0
            wfs_distances[i] = norm(zernikes[idx1,:] - zernikes[idx2,:])
        end
        
        embed_dim_distances[k][i] = norm(model(lanterns[idx1,:]) - model(lanterns[idx2,:]))
    end
end

f = Figure()
begin
    ax = Axis(f[1,1], xlabel="Euclidean distance of WFS image pairs", ylabel="Euclidean distance of embedded representation of PL images")
    for (k, c) in zip([5, 9, 18], [:deepskyblue, :forestgreen, :orange])
        CairoMakie.scatter!(ax, wfs_distances, embed_dim_distances[k], label="#Hidden layers = $k", markersize=3, color=c)
    end
    axislegend(; position=:lt)
    save("figures/embed_dim_pl_plumes.pdf", f)
    save("figures/embed_dim_pl_plumes.png", f)
    f
end

begin
    f = Figure(fonts=(;regular = "CMU Bright"))
    ax = Axis(f[1,1], xlabel="Hidden layer size", ylabel="Loss (PL intensity)")
    lines!(ax, reverse(npzread("data/pl_nn/nlpca_identity_losses.npy")))
    save("figures/nlpca_identity_losses.pdf", f)
    save("figures/nlpca_identity_losses.png", f)
    f
end