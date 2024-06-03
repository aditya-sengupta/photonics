using Plots
using NPZ
default(fontfamily= "Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true)

amps = npzread("data/sweep_testset_amplitudes_.npy")
sweep = npzread("data/linear_sweeps/gs.npy")

begin
    nmodes = 9
    pl = []
    for k in 1:nmodes
        lin_k = sweep[k,:,:]
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright, aspect_ratio=:equal, xlim=(-1,1), ylim=(-1,1))
        plot!(amps, amps, ls=:dash, color=:black)
        plot!(amps, lin_k, alpha=a)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(900,900), dpi=600, suptitle="Photonic lantern Gerchberg-Saxton reconstruction, simulated")
    Plots.savefig("figures/gs_linearity_sim.pdf")
    p
end