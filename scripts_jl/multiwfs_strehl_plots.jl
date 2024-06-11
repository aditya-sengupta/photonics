using NPZ
using Plots
using Plots.PlotMeasures: mm
using LaTeXStrings
using StatsBase: mean

gr()
strehls = npzread("./data/strehls_grid_240610.npy")
strehls_64 = npzread("./data/strehls_grid_240610_64.npy")
strehls[7,:,:] .= strehls_64[1,:,:]
times = 0:(1/799):1
ax = axes(strehls)

default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box, label=nothing, grid=true)

cmap = Dict(1 => 2, 2 => 13, 3 => 12, 4 => 11)
namemap = ["No lantern (NCPA)", "Interaction matrix", "Neural network","Gerchberg-Saxton"]

begin
    pl = []
    for i in ax[1]
        p = plot(ylim=(0,1), title=L"$D/r_0$" * "= $(2^(i-1))", legend=false)
        for j in ax[2]
            plot!(times, strehls[i,j,:], color=cmap[j], alpha=(j == 5 ? 0.5 : 1))
        end
        push!(pl, p)
    end
    push!(pl, plot((1:4)', legend=true, framestyle=:none, label=["No lantern (NCPA)" "Interaction matrix" "Neural network" "Gerchberg-Saxton"], legendfontsize=12, color=[2 13 12 11]))
    pf = plot(pl..., layout=(2,4), size=(900,500), suptitle="Strehl ratio over time (s) for varying reconstructors and atmospheres")
    Plots.savefig("figures/strehl_over_time.pdf")
    pf
end

default(linewidth=3)

begin
    D_over_r0s = 2 .^ (0:6)
    p = plot(xticks=(0:6, D_over_r0s), xlabel=L"$D/r_0$", ylabel="Averaged Strehl ratio, last 200 frames")
    for j in ax[2]
        mean_strehls = hcat(mean(strehls[:,j,600:800], dims=2)...)'
        plot!(0:6, mean_strehls, label=namemap[j], color=cmap[j])
        scatter!(0:6, mean_strehls, label=nothing, color=cmap[j])
    end
    Plots.savefig("figures/eventual_strehl.pdf")
    p
end