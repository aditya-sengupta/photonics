using NPZ
using MultivariateStats
using Plots
using HDF5
using PhotonicLantern

f = 5
lantern_intensities_train = transpose(abs2.(
    npzread("data/sim_trainsets/fnumber/sim_trainset_lanterns_fnumber_$(f)_date_240827.npy")
))
lantern_intensities_test = transpose(abs2.(
    npzread("data/sim_trainsets/fnumber/sim_testset_lanterns_fnumber_$(f)_date_240827.npy")
))

M = fit(PCA, lantern_intensities_train);
Ytest = predict(M, lantern_intensities_test)
heatmap(reconstruct(M, Ytest))
heatmap(lantern_intensities_test)

function hexagon_pattern(nrings, core_offset)
    nports = 1 + 3 * nrings * (nrings - 1)
    port_positions = zeros(nports, 2)
    nports_so_far = 0
    for i in 0:(nrings-1)
        nports_per_ring = max(1, 6 * i)
        theta = 0.0
        current_position = i * core_offset * [cos(theta), sin(theta)]
        next_position = i * core_offset * [cos(theta + π / 3), sin(theta + π / 3)]
        for j in 0:(nports_per_ring-1)
            if i > 0 && j % i == 0
                theta += π / 3
                current_position = next_position
                next_position = i * core_offset * [cos(theta + π / 3), sin(theta + π / 3)]
            end
            if i == 0
                cvx_coeff = 0.0
            else
                cvx_coeff = (j % i) / i
            end
            port_positions[nports_so_far + 1, :] = (1 - cvx_coeff) * current_position + cvx_coeff * next_position
            nports_so_far += 1
        end
    end
    return port_positions
end

centroids = hexagon_pattern(3, 1)

Plots.scatter(centroids[:,1], centroids[:,2], aspect_ratio=:equal, legend=nothing)

launch_fields = abs2.(Array(h5open("data/test_spie24_design.hdf5")["plotting_launch_fields"]))

evs = eigvals(M)
evs ./= evs[1]

begin
    f = Figure(size=(600,600), suptitle="x")
    
    for i in 1:4, j in 1:4
        ga = f[i, j]
        ax = Axis(ga, xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false, title="$(round(evs[4*(i-1)+j], digits=4))")
        lantern_plot = sum(M.proj[k,4*(i-1)+j] * launch_fields[:,:,k] for k in 1:19)
        lantern_plot = 2 * (lantern_plot) ./ (maximum(lantern_plot) - minimum(lantern_plot))
        lantern_plot .-= mean(lantern_plot)
        hm = CairoMakie.heatmap!(ax, lantern_plot, colormap=:seismic, colorrange=(-1, 1))
    end
    save("figures/pca_lantern_patterns.png", f)
    f
end

