using NPZ
using Plots
using PhotonicLantern

p_i_all = npzread("data/pl_231130/linearity_intensities.npy")
amps = npzread("data/pl_231130/linearity_amps.npy")

push_ref = 0.1
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

shane_modes = ["focus", "astig", "astig45", "tricoma", "tricoma60", "coma", "coma90", "spherical44-", "spherical44+", "spherical40", "spherical42+", "spherical42-"]

amperes2rad = 2π * 0.51255 / 1.55  

begin
    nmodes = 12
    pl = []
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        slope = 1 # (lin_k[r,k-1] - lin_k[l,k-1]) / (2 * push_ref)
        lin_k_primary = lin_k[:,k-1]
        deviation = abs.(amps * slope ./ lin_k_primary .- 1)
        linear_start = findfirst(deviation .< 0.2)
        linear_end = findlast(deviation .< 0.2)
        # lr = amps[linear_end] - amps[linear_start]
        pk = plot(xlabel=shane_modes[k-1], label=nothing, legend=:outertopright)
        plot!(amps * amperes2rad, lin_k * amperes2rad, alpha=a)
        plot!(amps * amperes2rad, amps * slope * amperes2rad, ls=:dash, color=:black)
        # vspan!([amps[linear_start], amps[linear_end]], alpha=0.2)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, suptitle="ShaneAO linearity plots (rad), 2023-11-30")
    Plots.savefig("figures/shane_1130_linearity.png")
    p
end