using NPZ
using Plots
using PhotonicLantern

default(fontfamily="Computer Modern", linewidth=3, framestyle=:box, label=nothing, grid=true)

amps = npzread("data/sweep_testset_amplitudes_.npy")
p_i_all = npzread("data/sweep_testset_lanterns.npy") .|> abs2

command_matrix = make_command_matrix(amps, p_i_all, push_ref=0.05, nmodes=18)
# plot_linearity(amps, p_i_all, command_matrix)

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

mode_names = ["x-tilt", "y-tilt", "astig", "focus", "astig45", "tricoma", "tricoma60", "coma", "coma90", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

begin
    nmodes = 9
    pl = []
    for k in 1:nmodes
        lin_k = ((p_i_all[k,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel=mode_names[k], label=nothing, legend=:outertopright, aspect_ratio=:equal, xlim=(-1,1), ylim=(-1,1))
        plot!(amps, amps, ls=:dash, color=:black)
        plot!(amps, lin_k, alpha=a)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(900,900), dpi=600, suptitle="Photonic lantern linear reconstruction, simulated")
    Plots.savefig("figures/linear_sim.pdf")
    p
end

sweep = zeros(nmodes, size(p_i_all, 2), nmodes);
for k in 1:nmodes
    sweep[k,:,:] = (p_i_all[k,:,:] * iA)[:,1:nmodes]
end
npzwrite("data/linear_sweeps/pl_linear.npy", sweep)