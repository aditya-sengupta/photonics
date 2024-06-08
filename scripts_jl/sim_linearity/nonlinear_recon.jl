using NPZ
using Plots
using Chain: @chain
using Flux
using JLD2

pgfplotsx()

default(fontfamily= "Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true, legend_font_halign=:left)

zernike_inputs = npzread("data/nonlinear_test/zernike_inputs.npy")
output_coeffs = abs2.(npzread("data/nonlinear_test/output_coeffs.npy"))
rms_pyramid = abs2.(npzread("data/nonlinear_test/pyramid_rms_residuals.npy"))
rms_inputs = npzread("data/nonlinear_test/rms_inputs.npy")
rms_residuals = npzread("data/nonlinear_test/rms_residuals.npy")
gs_rms_residuals = npzread("data/nonlinear_test/gs_rms_residuals.npy")

zero_one_ify(x) = (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin

timestamp = "240428_1706"
trainset_amplitudes, xmin, xmax = @chain npzread("data/sim_trainsets/sim_trainset_amplitudes_$timestamp.npy") transpose Matrix zero_one_ify
trainset_lanterns, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_$timestamp.npy") abs2.(_) transpose Matrix zero_one_ify

outputs = @. (output_coeffs - ymin) / (ymax - ymin)

model_state = JLD2.load("data/pl_nn/pl_nn_0.25.jld2", "model_state");
model = Flux.Chain(
    Dense(19 => 2000, relu),
    Dense(2000 => 100, relu),
    Dense(100 => 9)
)
Flux.loadmodel!(model, model_state)

function rearrange(x)
    y = copy(x)
    y[3], y[4] = y[4], y[3]
    y
end

nn_results = rescale.(hcat([model((x)) for x in eachrow(outputs)]...)', xmin, xmax)
nn_resids = nn_results .- zernike_inputs
nn_resids_rms = [sqrt(sum(x .^ 2)) for x in eachrow(nn_resids)]

begin
    msv = 2
    s = scatter(rms_inputs, rms_pyramid, msa=0, ms=msv, label="Modulated pyramid", xlabel="Injected aberration RMS (rad)", ylabel="Reconstruction error RMS (rad)", legend=:topleft, legendfontsize=12, color=14)
    scatter!(rms_inputs, rms_residuals, msa=0, ms=msv, label="PL linear reconstructor", color=13)
    scatter!(rms_inputs, nn_resids_rms, msa=0, ms=msv, label="PL neural network", color=12, alpha=0.5)
    scatter!(rms_inputs, gs_rms_residuals, msa=0, ms=msv, label="PL Gerchberg-Saxton", color=11, alpha=0.5)
    Plots.savefig("figures/nonlinear_recon.pdf")
    s
end