using NPZ
using Plots
using PhotonicLantern
default(fontfamily= "Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true)

p_i_all = npzread("data/pl_231025/linearity_responses.npy")# * rad2opd
amps = npzread("data/pl_231025/linearity_amps.npy")# * rad2opd

command_matrix = make_command_matrix(amps, p_i_all, nmodes=9, push_ref=0.05)
# plot_linearity(amps, p_i_all, command_matrix, nmodes=9)
begin
    nmodes = 9
    pl = []
    for k in 1:nmodes
        lin_k = ((p_i_all[k,:,:]) * command_matrix)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright, aspect_ratio=:equal, ylim=(-1, 1), xlim=(-1, 1))
        plot!(amps, amps, ls=:dash, color=:black)
        plot!(amps, lin_k, alpha=a)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(900,900), dpi=600, suptitle="Photonic lantern linear reconstruction, SEAL")
    Plots.savefig("figures/linear_seal.pdf")
    p
end
