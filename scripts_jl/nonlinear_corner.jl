using NPZ
using CairoMakie
using StatsBase
using ProgressMeter
X_pred = npzread("data/X_pred_larger.npy")
X_test = npzread("data/X_test_m_larger.npy")

znames = ["x-tilt", "y-tilt", "focus", "astig", "astig45", "coma", "coma90", "tricoma", "tricoma60", "spherical44-",  "spherical42-", "spherical40", "spherical42+", "spherical44+"]
append!(znames, ["Z$i" for i in 15:19])

hist_w = nothing
begin
    n = 19
    f = Figure(size = (100n, 100n))

    @showprogress for i in 1:n, j in 1:n
        ga = f[i, j]
        ax = Axis(ga, xticksvisible=false, yticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        hist_w = fit(Histogram, (X_pred[i,:], X_test[j,:]), (-0.5:0.01:0.5001, -0.5:0.01:0.5001))
        hm = CairoMakie.heatmap!(ax, sqrt.(hist_w.weights), colormap=(i == j ? :summer : :viridis))
        if j == 1
            ax.ylabel = "rec " * znames[i]
        end
        if i == n
            ax.xlabel = "inj " * znames[j]
        end
    end

    f
end
save("figures/inj_rec_correlations_larger.pdf", f)

cov_m = reverse(cor(vcat(X_pred, X_test)')[1:19,20:38], dims=1)

begin
    f2 = Figure()
    ax2 = Axis(f2[1,1])
    CairoMakie.heatmap!(ax2, cov_m)
    Colorbar(f2[1,2])
    f2
end

