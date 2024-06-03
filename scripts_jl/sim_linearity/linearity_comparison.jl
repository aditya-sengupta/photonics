using Plots
using NPZ
default(fontfamily= "Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true)

amps = npzread("data/sweep_testset_amplitudes_.npy")
sweep_gs = npzread("data/linear_sweeps/gs.npy")
sweep_mpwfs = npzread("data/linear_sweeps/mpwfs.npy")
sweep_linear = npzread("data/linear_sweeps/pl_linear.npy")
sweep_nn = npzread("data/linear_sweeps/pl_nn.npy")

begin
    nmodes = 9
    pl = []
    for k in 1:nmodes
        pk = plot(xlabel=mode_names[k], legend=:bottomright, aspect_ratio=:equal, xlim=(-1,1), ylim=(-1,1))
        plot!(amps, amps, ls=:dash, color=:black)
        for (i, (s, n)) in enumerate(zip([sweep_mpwfs, sweep_linear, sweep_nn, sweep_gs], ["Modulated pyramid", "PL interaction matrix", "PL neural network", "PL Gerchberg-Saxton"]))
            plot!(amps, s[k,:,k], color=15-i, label=(k == 3 ? n : nothing))
        end
        push!(pl, pk)
    end
    p = plot(pl..., size=(1000,1000), dpi=600, suptitle="All PL reconstruction algorithms compared with the modulated pyramid")
    Plots.savefig("figures/all_recon.pdf")
    p
end