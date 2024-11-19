using NPZ
using StatsBase: norm
using CairoMakie

zernikes = npzread("data/sim_trainsets/fnumber/sim_trainset_amplitudes_fnumber_3_date_240827.npy")
lanterns = abs2.(npzread("data/sim_trainsets/fnumber/sim_trainset_lanterns_fnumber_3_date_240827.npy"))

begin
    f = Figure()
    ax = Axis(f[1,1], xlabel="Euclidean distance of WFS image pairs", ylabel="Euclidean distance of PL image pairs")
    for (k, c) in zip([5, 9, 18], [:deepskyblue, :forestgreen, :orange])
        ntrials = 10_000
        wfs_distances = zeros(ntrials)
        pli_distances = zeros(ntrials)
        sign_inversions = zeros(ntrials)
        for i in 1:ntrials
            # draw two random integers to use as the indices for this trial
            idx1 = rand(1:size(zernikes,1))
            idx2 = rand(1:size(zernikes,1))
            
            # compute the wavefront sensor (WFS) distance
            zernikes[idx1]
            wfs_distances[i] = norm(zernikes[idx1,1:k] - zernikes[idx2,1:k])
            sign_inversions[i] = sum(sign.(zernikes[idx1,1:k]) .!= sign.(zernikes[idx2,1:k]))
            
            # compute the photonic lantern image (PLI) distance
            pli_distances[i] = norm(lanterns[idx1,:] - lanterns[idx2,:])
        end

        CairoMakie.scatter!(ax, wfs_distances, pli_distances, label="#Zernikes = $k", markersize=3, color=c)
    end
    axislegend(; position=:lt)
    save("figures/phase_pl_plumes.pdf", f)
    save("figures/phase_pl_plumes.png", f)
    f
end

