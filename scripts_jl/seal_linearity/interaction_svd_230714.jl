using NPZ
using Plots
using LinearAlgebra: svd, Diagonal
using ZernikePolynomials
using PhotonicLantern: im_show, nanify
using Measures

zv = 2:19

amps = npzread("data/pl_230714/linearity_amplitudes_230714_1645.npy")
p_i_all = npzread("data/pl_230714/linearity_intensities_230714_1645.npy")
push_ref = 0.1
l, r = argmin(abs.(amps .+ push_ref)), argmin(abs.(amps .- push_ref))
p_i = p_i_all[:,[l,r],:]
s1, s2, s3 = size(p_i_all, 1), size(p_i, 2), size(p_i, 3)
z_i = zeros(s1, 2, s1)
for k in 1:s1
    z_i[k,:,k] = [-push_ref, push_ref]
end

powers = reshape(p_i, ((s1 * s2, s3)))
z = reshape(z_i, ((s1 * s2, s1)))

A = z \ powers
U, S, Vt = svd(A)

begin
    lantern_basis = []
    for row in eachrow(inv(U))
        push!(lantern_basis, evaluateZernike(256, collect(zv), row))
    end
end

begin
    pl = []
    for (basis_element, s) in zip(lantern_basis, S)
        push!(pl, im_show(basis_element, legend=:none, margin=-10.0mm, title=string(round(s, digits=3)), titlefontsize=8))
    end
    plot(pl..., layout=(3,6), margin=-1.0mm, plot_title="Principal lantern modes with singular values", plot_titlefontsize=13)
end

port_positions = npzread("data/pl_230525/seal_port_positions.npy")

begin
    pl = []
    m = maximum(U * Diagonal(S))
    for (col, s) in zip(eachcol(Vt), S)
        push!(
            pl,
            scatter(port_positions[1,:], port_positions[2,:], msw=0, aspect_ratio=:equal, legend=nothing, grid=nothing, showaxis=false, ms=2, alpha=clamp.((col * s) ./ m, 0.2, 1))
        )
    end
    plot(pl..., layout=(3,6), plot_title="Principal lantern responses", plot_titlefontsize=13)
end

iA = p \ z

U, S, Vt = svd(A)
iU, iS, iVt = svd(iA)

z * U

begin
    nmodes = 18
    pl = []
    transformed_intensities = copy(p_i_all)
    for (i, a) in enumerate(amps)
        transformed_intensities[:,i,:] = transformed_intensities[:,i,:] * U
    end
    for k in 1:(nmodes)
        lin_k = ((transformed_intensities[k,:,:]) * iU * Diagonal(iS))[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 1:(nmodes)]'
        pk = plot(xlabel="f$(k)", label=nothing, legend=:outertopright, xticks=nothing, yticks=nothing)
        plot!(amps, lin_k, alpha=a)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200, layout=(3, 6))
    Plots.savefig("figures/linear_svd_230714.png")
    p
end
