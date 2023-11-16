using NPZ
using Plots
using LinearAlgebra: svd

# cd("photonics")

λ = 635.0 # nm
rad2opd = λ / (2pi)

p_i_all = reduce((x,y) -> cat(x, y, dims=3), [npzread("data/pl_231028/z$x.npy") for x in 1:6]')
amps = collect(-1.0:0.2:1.0)

push_ref = 0.2
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
    nmodes = 6
    pl = []
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        lin_k_primary = lin_k[:,k-1]
        pk = plot(xlabel="Mode $(k)", label=nothing, legend=:outertopright)
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps, ls=:dash, color=:black)
        push!(pl, pk)
    end
    p = plot(pl..., legend=nothing, size=(750,500), dpi=200)
    Plots.savefig("figures/shane_daytime_linearity.png")
    p
end


