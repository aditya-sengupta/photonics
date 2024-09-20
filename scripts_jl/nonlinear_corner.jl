using NPZ
using CairoMakie
using StatsBase
using ProgressMeter
X_pred = npzread("data/X_pred.npy")
X_test = npzread("data/X_test_m.npy")

znames = ["tip", "tilt", "focus", "astig", "astig45", "coma90", "coma", "tricoma90", "tricoma"]
append!(znames, ["Z$i" for i in 10:19])

hist_w = nothing
begin
    n = 19
    f = Figure(size = (100n, 100n))

    @showprogress for i in 1:n, j in 1:n
        ga = f[i, j]
        ax = Axis(ga, xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        hist_w = fit(Histogram, (X_pred[i,:], X_test[j,:]), (-0.25:0.005:0.2501, -0.25:0.005:0.2501))
        hm = heatmap!(ax, hist_w.weights, colormap=:viridis)
        if j == 1
            ax.ylabel = "inj " * znames[i]
        end
        if i == n
            ax.xlabel = "rec " * znames[j]
        end
    end

    f
end


cov_m = reverse(cor(vcat(X_pred, X_test)')[1:19,20:38], dims=1)

begin
    f2 = Figure()
    ax2 = Axis(f2[1,1])
    heatmap!(ax2, cov_m_masked)
    Colorbar(f2[1,2])
    f2
end

save("t.pdf", f)