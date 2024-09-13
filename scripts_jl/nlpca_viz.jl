using NPZ
using Plots

d = []
for f in 3:11
    push!(d, npzread("data/pl_nn/test_losses_noisy_19_f_$f.npy") |> reverse)
end
d = hcat(d...)

heatmap(3:11, 0:20, d, xticks=3:11, xlabel="f/#", ylabel="Hidden layer dimension", title="PL RMS error with photon noise", c=:summer)