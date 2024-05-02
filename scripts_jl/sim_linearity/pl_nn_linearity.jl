using Chain
using NPZ
using JLD2
using Flux
using Plots

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin
amplitudes = npzread(pwd() * "/data/sweep_testset_amplitudes_.npy")
sweep_lanterns = abs2.(npzread(pwd() * "/data/sweep_testset_lanterns.npy"))

lim_to_tstamp = Dict(0.12 => 1702, 0.25 => 1706, 0.5 => 1712, 1.0 => 1716)

for lim in keys(lim_to_tstamp)
    tstamp = lim_to_tstamp[lim]
    _, xmin, xmax = @chain npzread(pwd() * "/data/sim_trainsets/sim_trainset_amplitudes_240428_$tstamp.npy") transpose Matrix zero_one_ify
    _, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_240428_$tstamp.npy") abs2.(_) transpose Matrix zero_one_ify

    model_state = JLD2.load(pwd() * "/data/pl_nn/pl_nn_$lim.jld2", "model_state");
    model = Flux.Chain(
        Dense(19 => 2000, relu),
        Dense(2000 => 100, relu),
        Dense(100 => 9)
    )
    Flux.loadmodel!(model, model_state)
    sweep_lanterns_normalized = (sweep_lanterns .- ymin) ./ (ymax - ymin) .|> Float32

    mode_names = ["x-tilt", "y-tilt", "astig", "focus", "astig45", "tricoma", "tricoma60", "coma", "coma90"]

    nmodes = 9
    pl = []
    for k in 1:nmodes
        nn_recon = rescale.(hcat([model(x) for x in eachrow(sweep_lanterns_normalized[k,:,:])]...)', xmin, xmax)
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright)
        plot!(amplitudes, nn_recon, alpha=a)
        plot!(amplitudes, amplitudes, ls=:dash, color=:black)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,750), dpi=600, suptitle="NN recon. simulation, max train amplitude = $lim rad")
    Plots.savefig(pwd() * "/figures/nn_sim_$lim.png")
    p
end