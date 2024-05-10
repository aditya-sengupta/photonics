using Chain: @chain
using NPZ
using JLD2
using Flux
using Plots

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin
amplitudes = npzread("data/sweep_testset_amplitudes_19.npy")
sweep_lanterns = abs2.(npzread("data/sweep_testset_lanterns_19.npy"))

_, xmin, xmax = @chain npzread("data/sim_trainsets/sim_trainset_amplitudes_240502_1947.npy") transpose Matrix zero_one_ify
_, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_240502_1947.npy") abs2.(_) transpose Matrix zero_one_ify

model_state = JLD2.load("data/pl_nn/pl_nn_0.5_19.jld2", "model_state");
model = Flux.Chain(
    Dense(19 => 2000, relu),
    Dense(2000 => 100, relu),
    Flux.Dropout(0.2),
    Dense(100 => 19)
)
Flux.loadmodel!(model, model_state)
sweep_lanterns_normalized = (sweep_lanterns .- ymin) ./ (ymax - ymin) .|> Float32

mode_names = ["x-tilt", "y-tilt", "astig", "focus", "astig45", "tricoma", "tricoma60", "coma", "coma90", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

nmodes = 18
pl = []
for k in 1:nmodes
    nn_recon = rescale.(hcat([model(x) for x in eachrow(sweep_lanterns_normalized[k,:,:])]...)', xmin, xmax)
    a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
    pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright)
    plot!(amplitudes, nn_recon, alpha=a)
    plot!(amplitudes, amplitudes, ls=:dash, color=:black)
    push!(pl, pk)
end
p = plot(pl..., legend=nothing, size=(1500,750), dpi=600, suptitle="NN recon. simulation, max train amplitude = 0.5 rad", layout=(3, 6))
Plots.savefig("figures/nn_sim_0.5_19.png")
p
