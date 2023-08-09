using NPZ
using Plots

#z_i = npzread("data/pl_230713/inputzs_230713_1314.npy")
#p_i = npzread("data/pl_230713/pls_230713_1314.npy")
#s = sum(p_i)
#p_i ./= s

p_i_all = npzread("data/pl_230713/pl_linear_intensities.npy")
p_i = npzread("data/pl_230713/pl_linear_intensities.npy")[:,[10,12],:]
s1, s2, s3 = size(p_i, 1), size(p_i, 2), size(p_i, 3)
z_i = zeros(s1, 2, s1)
amp = collect(-1:0.1:1)
for k in 1:5
    z_i[k,:,k] = [-0.1, 0.1]
end

p = reshape(p_i, ((s1 * s2, s3)))
z = reshape(z_i, ((s1 * s2, s1)))

A = z \ p
iA = p \ z
resid = p - z * A
B = z \ sqrt.(resid)
iB = sqrt.(resid) \ z
resid2 = sqrt.(resid) - z * B
C = z \ cbrt.(resid2)
iC = cbrt.(resid2) \ z

begin
    nmodes = 5
    pl = []
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * iA)[:,1:nmodes]
        a = [(k == i ? 1 : 0.3) for i in 2:(nmodes+1)]'
        push!(pl, plot(amp, lin_k, xlabel="injected ($(k))", ylabel="recovered", label=nothing, legend=:outertopright, alpha=a))
    end
    p = plot(pl..., label=["2" "3" "4" "5" "6"], size=(750,500), dpi=200)
    Plots.savefig("figures/quadratic_230713.png")
    p
end