using NPZ
using Plots
using PhotonicLantern

k = 120
multiwfs_psfs = npzread("data/multiwfs_psfs.npy")[:,k+1:end-k,k+1:end-k]
r = 1:2:size(multiwfs_psfs, 1)
sr = npzread("data/multiwfs_anim_strehls.npy")

anim = @animate for i ∈ r
    t = i > 100 ? "pyramid + lantern" : "pyramid"
    im_show(log10.(multiwfs_psfs[i,:,:] ./ maximum(multiwfs_psfs[i,:,:])), cbar=nothing, title="Strehl = $(round(100*sr[i], digits=1))%, $t", clim=(-5, 0), c=:inferno, dpi=600)
end
gif(anim, "figures/psf_cl.gif", fps = 5)
