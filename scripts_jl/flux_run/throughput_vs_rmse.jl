using NPZ
using Plots
using PhotonicLantern

fs = npzread("data/pl_nn/fnumbers_noisy_maxperf_19.npy")
losses = npzread("data/pl_nn/test_losses_noisy_maxperf_19.npy")
throughput_fs = npzread("data/throughput_fs.npy")
throughputs = npzread("data/throughputs.npy")

begin
    plot(throughput_fs[1:80], throughputs[1:80], label="Throughput", xlabel="f/#", ylabel="PL throughput (%)")
    plot!([5], [0.1], label="RMSE", color=2)
    plot!(twinx(), fs, sqrt.(losses), color=2, label=nothing, ylabel="Best test-set RMS error (rad)")
end