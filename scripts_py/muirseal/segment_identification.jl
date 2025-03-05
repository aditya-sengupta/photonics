using NPZ
using Plots
using StatsBase: std

flat = float.(npzread("data/pl_250117/flat.npy"))

@gif for i in 1:167
    heatmap(float.(npzread("data/pl_250117/$(i)_tip.npy")) - flat, clim=(-1000,1000), title="$(i)_tip", c=:diverging_bkr_55_10_c35_n256)
end

begin
    plot([std(float.(npzread("data/pl_250117/$(i)_piston.npy")) - flat) for i in 1:167], xlabel="Segment number", ylabel="Standard deviation of difference image", label="Piston")
    plot!([std(float.(npzread("data/pl_250117/$(i)_tip.npy")) - flat) for i in 1:167], label="Tip")
    plot!([std(float.(npzread("data/pl_250117/$(i)_tilt.npy")) - flat) for i in 1:167], label="Tilt")
end

function hexagon_pattern(nrings, core_offset)
    nports = 1 + 3 * nrings * (nrings - 1)
    port_positions = zeros(nports, 2)
    nports_so_far = 0
    for i in 0:(nrings-1)
        nports_per_ring = max(1, 6 * i)
        theta = -π/3
        current_position = i * core_offset * [cos(theta), sin(theta)]
        next_position = i * core_offset * [cos(theta + π / 3), sin(theta + π / 3)]
        for j in 0:(nports_per_ring-1)
            if i > 0 && j % i == 0
                theta += π / 3
                current_position = next_position
                next_position = i * core_offset * [cos(theta + π / 3), sin(theta + π / 3)]
            end
            cvx_coeff = i == 0 ? 0.0 : (j % i) / i
            port_positions[nports_so_far + 1, :] = (1 - cvx_coeff) * current_position + cvx_coeff * next_position
            nports_so_far += 1
        end
    end
    return port_positions
end

begin
    sc = [std(float.(npzread("data/pl_250117/$(i)_tip.npy")) - flat) for i in 1:167]
    push!(sc, 0)
    push!(sc, 0)
end

begin
    pattern = hexagon_pattern(8, 1.0)
    scatter(pattern[:,1], pattern[:,2], zcolor=sc, aspect_ratio=:equal, legend=false, marker=:circle, axis=nothing)
end

