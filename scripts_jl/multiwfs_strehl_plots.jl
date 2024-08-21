using NPZ
using Plots
using Plots.PlotMeasures: mm
using LaTeXStrings
using StatsBase: mean
using ColorSchemes

csch = eval(Meta.parse("ColorSchemes." * string(cvdschemes[53])))

strehls = npzread("./data/strehls_grid_240610.npy")
strehls_64 = npzread("./data/strehls_grid_240610_64.npy")
strehls[7,:,:] .= strehls_64[1,:,:]
times = 0:(1/799):1
ax = axes(strehls)

default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box, label=nothing, grid=true)

namemap = ["No lantern (NCPA)", "Linear reconstructor", "Neural network","Gerchberg-Saxton"]

begin
    gr()
    pl = []
    for i in ax[1]
        p = plot(ylim=(0,1), title=L"$D/r_0$" * "= $(2^(i-1))", legend=false)
        vline!([0.125], ls=:dash, color=:black)
        for j in ax[2]
            plot!(times, strehls[i,j,:], color=csch[2j-1], alpha=(j == 5 ? 0.5 : 1))
        end
        push!(pl, p)
    end
    push!(pl, plot((1:5)', legend=true, framestyle=:none, label=["No lantern (NCPA)" "Linear reconstructor" "Neural network" "Gerchberg-Saxton" "PL loop closed"], legendfontsize=12, color=[csch[1] csch[3] csch[5] csch[7] :black], ls=[:solid :solid :solid :solid :dash]))
    pf = plot(pl..., layout=(2,4), size=(900,500), suptitle="Strehl ratio over time (s) for varying reconstructors and atmospheres")
    Plots.savefig("figures/strehl_over_time.pdf")
    pf
end

markertypes = [:rect, :diamond, :circle, :rtriangle]

begin
    pgfplotsx()
    D_over_r0s = 2 .^ (0:6)
    p = plot(xticks=(0:6, D_over_r0s), xlabel=L"$D/r_0$", ylabel="Average Strehl ratio after lantern loop closed")
    for j in ax[2]
        mean_strehls = hcat(mean(strehls[:,j,400:800], dims=2)...)'
        plot!(0:6, mean_strehls, color=csch[2j-1], label=nothing)
        scatter!(0:6, mean_strehls, color=csch[2j-1], markershape=markertypes[j], label=namemap[j], markersize=(j == 2 ? 7 : 4), legend=:bottomleft, legendfontsize=14)
    end
    Plots.savefig("figures/eventual_strehl.pdf")
    p
end
