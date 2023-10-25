using NPZ
using StatsBase: mean
using Plots
using InvertedIndices
using MultivariateStats
using ZernikePolynomials

rms(x) = sqrt(sum(x .^ 2))

improvements = vcat(npzread("./data/pl_231020/rms_improvements_100.npy"), npzread("./data/pl_231020/rms_improvements_100_t2.npy"))
input_amps = vcat(npzread("./data/pl_231020/before_amps_100.npy"), npzread("./data/pl_231020/before_amps_100_t2.npy"))
output_reads = vcat(npzread("./data/pl_231020/after_reads_100.npy"), npzread("./data/pl_231020/after_reads_100_t2.npy"))

failure_locations = findall(improvements .> 1)
success_init_rmsvals = vec(sqrt.(mean((input_amps[Not(failure_locations), :]).^2, dims=2)))
failure_init_rmsvals = vec(sqrt.(mean((input_amps[failure_locations, :]).^2, dims=2)))

histogram(improvements, bins=50, legend=nothing, title="Photonic lantern CL/OL RMS, 10 modes, 0.1 rad", xlabel="Closed-loop RMS / open-loop RMS", ylabel="Count", lw=0, la=1)

histogram(success_init_rmsvals)
histogram!(failure_init_rmsvals)

rms_before = vec(sqrt.(sum(input_amps .^ 2, dims=2)))
rms_after = vec(sqrt.(sum(output_reads .^ 2, dims=2)))

begin
    histogram(rms_before, label="Injected amplitudes", xlim=(0.0, 0.5), ylim=(0, 40), xlabel="RMS error (rad)", ylabel="Count", bar_width=0.02, lw=0)
    histogram!(rms_after, bar_width=0.02, alpha=0.5, label="Closed-loop amplitudes", lw=0, color=3)
end

p = fit(PCA, input_amps[(failure_locations), :]')
highest_variance = projection(p)[:,1]
im_show(evaluateZernike(256, collect(2:12), highest_variance))
# fit(PCA, input_amps[Not(failure_locations), :]')

