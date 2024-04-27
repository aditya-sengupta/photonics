using NPZ
using Plots
using LinearAlgebra: svd

# cd("photonics")

λ = 635.0 # nm
rad2opd = λ / (2pi)

p_i_all = npzread("data/pl_231025/linearity_responses.npy")# * rad2opd
amps = npzread("data/pl_231025/linearity_amps.npy")# * rad2opd

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
    nmodes = 10
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
        pk = plot(xlabel="Mode $(k)", label=nothing, legend=:outertopright)
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps, ls=:dash, color=:black)
        # vspan!([amps[linear_start], amps[linear_end]], alpha=0.2)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200)
    Plots.savefig("figures/seal_linearity_extended.png")
    p
end

begin
    nmodes = 6
    pl = []
    for (k, name) in zip(2:(nmodes+1), ["x-tilt", "y-tilt", "focus", "astig", "astig45", "tricoma"])
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        slope = (lin_k[r,k-1] - lin_k[l,k-1]) / (2 * push_ref)
        lin_k_primary = lin_k[:,k-1]
        deviation = abs.(amps * slope ./ lin_k_primary .- 1)
        linear_start = findfirst(deviation .< 0.25)
        linear_end = findlast(deviation .< 0.25)
        lr = amps[linear_end] - amps[linear_start]
        pk = plot(xlabel=name, label=nothing, legend=:outertopright)
        plot!(amps, lin_k[:,[1, 2, 3, k-1]], alpha=a[:,[1,2,3,k-1]], c=[1 2 3 k-1])
        plot!(amps, amps, ls=:dash, color=:black)
        # vspan!([amps[linear_start], amps[linear_end]], alpha=0.2)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, layout=(2, 3), suptitle="There's a lot more tip-tilt cross-talk than higher orders")
    Plots.savefig("figures/linear_with_axes_231012.png")
    p
end

begin
    nmodes = 6
    pl = []
    for (k, name) in zip(2:(nmodes+1), ["x-tilt", "y-tilt", "focus", "astig", "astig45", "tricoma"])
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        lin_k_primary = lin_k[:,k-1]
        pk = plot(xlabel=name, label=nothing, legend=:outertopright)
        plot!(amps, abs.(lin_k[:,[1, 2, 3, k-1]] ./ lin_k[:,k-1]), c=[1 :blue 2 3])
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, layout=(2, 3), suptitle="There's a lot more tip-tilt cross-talk than higher orders")
    Plots.savefig("figures/tt_crosstalk_231012.png")
    p
end