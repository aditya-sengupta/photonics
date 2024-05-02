using NPZ
using Plots
using PhotonicLantern

# cd("photonics")

amps = npzread("data/sweep_testset_amplitudes_.npy")
p_i_all = npzread("data/sweep_testset_lanterns.npy") .|> abs2

command_matrix = make_command_matrix(amps, p_i_all, push_ref=0.05, nmodes=18)
plot_linearity(amps, p_i_all, command_matrix)

push_ref = 0.05
l, r = argmin(abs.(amps .+ push_ref)), argmin(abs.(amps .- push_ref))
p_i = p_i_all[:,[l,r],:]
s1, s2, s3 = size(p_i_all, 1), size(p_i, 2), size(p_i, 3)
z_i = zeros(s1, 2, s1)
for k in 1:s1
    z_i[k,:,k] = [-push_ref, push_ref]
end

p = reshape(p_i, ((s1 * s2, s3)))
z = reshape(z_i, ((s1 * s2, s1)))

A = z \ p
iA = p \ z
resid = p - z * A

begin
    nmodes = 9
    pl = []
    for k in 1:nmodes
        lin_k = ((p_i_all[k,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright)
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps, ls=:dash, color=:black)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,750), dpi=600, layout=(3, 3), suptitle="Linear reconstruction, simulated data")
    Plots.savefig("figures/linear_sim.png")
    p
end
