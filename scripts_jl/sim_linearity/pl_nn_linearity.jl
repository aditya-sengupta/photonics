using Chain: @chain
using NPZ
using JLD2
using Flux
using Plots
using Plots.PlotMeasures
using StatsBase: mean

default(fontfamily="Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true)

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin
amplitudes = npzread("data/sweep_testset_amplitudes_19.npy")
sweep_lanterns = abs2.(npzread("data/sweep_testset_lanterns_19.npy"))

timestamp = "spieeval"
trainset_amplitudes, xmin, xmax = @chain npzread("data/sim_trainsets/sim_trainset_amplitudes_$timestamp.npy") transpose Matrix zero_one_ify
trainset_lanterns, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_$timestamp.npy") abs2.(_) transpose Matrix zero_one_ify

trainset_lanterns = Float32.(trainset_lanterns)

model_state = JLD2.load("data/pl_nn/pl_nn_spieeval.jld2", "model_state");
model = Flux.Chain(
    Dense(19 => 2000, relu),
    Dense(2000 => 100, relu),
    Dense(100 => 9)
)
Flux.loadmodel!(model, model_state)
sweep_lanterns_normalized = (sweep_lanterns .- ymin) ./ (ymax - ymin) .|> Float32

mode_names = ["x-tilt", "y-tilt", "astig", "focus", "astig45", "tricoma", "tricoma60", "coma", "coma90", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

nmodes = 9
pl = []
for k in 1:nmodes
    nn_recon = rescale.(hcat([model(x) for x in eachrow(sweep_lanterns_normalized[k,:,:])]...)', xmin, xmax)
    a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
    pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright, ylim=(-1, 1), xlim=(-1,1))
    plot!(amplitudes, amplitudes, ls=:dash, color=:black)
    plot!(amplitudes, nn_recon, alpha=a)
    push!(pl, pk)
end
p = plot(pl..., legend=nothing, size=(900,900), dpi=600, suptitle="Neural network reconstructor simulation", layout=(3, 3), left_margin=2Plots.mm)
Plots.savefig("figures/nn_sim_spie.pdf")
p

sweep = zeros(nmodes, length(amplitudes), nmodes);
for k in 1:nmodes
    sweep[k,:,:] = nn_recon = rescale.(hcat([model(x) for x in eachrow(sweep_lanterns_normalized[k,:,:])]...)', xmin, xmax)
end
npzwrite("data/linear_sweeps/pl_nn.npy", sweep)

trainset_amplitudes_scaled_f32 = rescale.(trainset_amplitudes[:,48001:end], xmin, xmax)
resid_train = rescale.(hcat(model.(eachcol(trainset_lanterns[:,48001:end]))...), xmin, xmax) - trainset_amplitudes_scaled_f32
mean(sqrt.(sum(resid_train .^ 2, dims=1)))