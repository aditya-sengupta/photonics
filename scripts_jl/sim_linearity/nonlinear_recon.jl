using NPZ
using Plots
using Chain: @chain
using Flux
using JLD2
using StatsBase: mean
using ColorSchemes
csch = eval(Meta.parse("ColorSchemes." * string(cvdschemes[53])))

pgfplotsx()

default(fontfamily= "Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true, legend_font_halign=:left)

function zero_one_ify(x, xmin=nothing, xmax=nothing)
    if isnothing(xmin) && isnothing(xmax)
        xmin, xmax = minimum(x), maximum(x)
    end
    return (x .- minimum(x)) ./ (maximum(x) .- minimum(x)), minimum(x), maximum(x)
end

zernike_inputs = npzread("data/nonlinear_test/test_phase_screens.npy")
output_coeffs = abs2.(npzread("data/nonlinear_test/readings.npy"))
output_coeffs_unresidualized = npzread("data/nonlinear_test/readings_unresidualized.npy")
rms_pyramid = abs2.(npzread("data/nonlinear_test/pyramid_rms_residuals.npy"))
rms_inputs = npzread("data/nonlinear_test/rms_inputs.npy")
rms_residuals = npzread("data/nonlinear_test/rms_residuals.npy")
gs_rms_residuals = npzread("data/nonlinear_test/gs_rms_residuals.npy")

rescale(z, zmin, zmax) = z * (zmax - zmin) + zmin

timestamp = "spieeval"
trainset_amplitudes, xmin, xmax = @chain npzread("data/sim_trainsets/sim_trainset_amplitudes_$timestamp.npy") transpose Matrix zero_one_ify
trainset_lanterns, ymin, ymax = @chain npzread("data/sim_trainsets/sim_trainset_lanterns_$timestamp.npy") abs2.(_) transpose Matrix zero_one_ify

testset_amplitudes = @chain npzread("data/sim_trainsets/sim_testset_amplitudes_$timestamp.npy") transpose Matrix
testset_lanterns = @chain npzread("data/sim_trainsets/sim_testset_lanterns_$timestamp.npy") abs2.(_) transpose Matrix zero_one_ify(_, ymin, ymax) (_[1])

outputs = @. (output_coeffs_unresidualized - ymin) / (ymax - ymin)

model_state = JLD2.load("data/pl_nn/pl_nn_spieeval.jld2", "model_state");
model = Flux.Chain(
    Dense(19 => 2000, relu),
    Dense(2000 => 100, relu),
    Dense(100 => 9)
)
Flux.loadmodel!(model, model_state)

nn_results = (model(outputs') .* (xmax - xmin) .+ xmin)'
nn_resids = nn_results .- zernike_inputs
nn_resids_rms = [sqrt(sum(x .^ 2)) for x in eachrow(nn_resids)]

begin
    msv = 2
    s = scatter(rms_inputs, rms_pyramid, msa=0, ms=msv, label="Modulated pyramid", xlabel="Injected aberration RMS (rad)", ylabel="Reconstruction error RMS (rad)", legend=:topleft, legendfontsize=12, color=csch[1])
    scatter!(rms_inputs, rms_residuals, msa=0, ms=msv, label="PL linear reconstructor", color=csch[3])
    scatter!(rms_inputs, nn_resids_rms, msa=0, ms=msv, label="PL neural network", color=csch[5])
    scatter!(rms_inputs, gs_rms_residuals, msa=0, ms=msv, label="PL Gerchberg-Saxton", color=csch[7])
    Plots.savefig("figures/nonlinear_recon.pdf")
    s
end

rms_error = mean(sum(((model(outputs') .* (xmax - xmin) .+ xmin) .- zernike_inputs') .^ 2, dims=1))
