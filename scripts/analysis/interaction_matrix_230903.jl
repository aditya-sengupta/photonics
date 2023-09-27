using NPZ
using Plots
using LinearAlgebra: svd

# cd("photonics")

p = npzread("data/pl_shane_230903/interaction_intensities.npy")
z = npzread("data/pl_shane_230903/interaction_zernikes.npy")

A = z \ p
iA = p \ z
resid = p - z * A

begin
    nmodes = 12
    pl = []
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        slope = (lin_k[r,k-1] - lin_k[l,k-1]) / (2 * push_ref)
        lin_k_primary = lin_k[:,k-1]
        deviation = abs.(amps * slope ./ lin_k_primary .- 1)
        linear_start = findfirst(deviation .< 0.2)
        linear_end = findlast(deviation .< 0.2)
        lr = amps[linear_end] - amps[linear_start]
        pk = plot(xlabel="z$(k), $(round(lr, digits=3)) rad", label=nothing, legend=:outertopright, xticks=nothing, yticks=nothing)
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps * slope, ls=:dash, color=:black)
        vspan!([amps[linear_start], amps[linear_end]], alpha=0.2)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, layout=(3, 6))
    Plots.savefig("figures/linear_230714.png")
    p
end

begin
    nmodes = 6
    pl = []
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        slope = (lin_k[r,k-1] - lin_k[l,k-1]) / (2 * push_ref)
        lin_k_primary = lin_k[:,k-1]
        deviation = abs.(amps * slope ./ lin_k_primary .- 1)
        linear_start = findfirst(deviation .< 0.25)
        linear_end = findlast(deviation .< 0.25)
        lr = amps[linear_end] - amps[linear_start]
        pk = plot(xlabel="z$(k), $(round(lr, digits=3)) rad", label=nothing, legend=:outertopright)
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps * slope, ls=:dash, color=:black)
        vspan!([amps[linear_start], amps[linear_end]], alpha=0.2)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, layout=(2, 3), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    Plots.savefig("figures/linear_with_axes_230714.png")
    p
end



