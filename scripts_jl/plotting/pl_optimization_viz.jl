using Plots
using NPZ
using PhotonicLantern
using Plots.PlotMeasures
using Distributions

default(fontfamily="Computer Modern", linewidth=1.5, framestyle=:box, label=nothing, grid=true)

ntrim = 200

std_lantern_output = reverse(abs2.(npzread("data/finesst2025/std_lantern_output.npy")), dims=1)[ntrim:end-ntrim,ntrim:end-ntrim]
modeselective_lantern_output = reverse(abs2.(npzread("data/finesst2025/modeselective_lantern_output.npy")), dims=1)[ntrim:end-ntrim,ntrim:end-ntrim]
manufacturing_error_lantern_output = reverse(abs2.(npzread("data/finesst2025/manufacturing_error_lantern_output.npy")), dims=1)[ntrim:end-ntrim,ntrim:end-ntrim]
modeselective_lantern_tilt_output = reverse(abs2.(npzread("data/finesst2025/modeselective_lantern_tilt_output.npy")), dims=1)[ntrim:end-ntrim,ntrim:end-ntrim]

propmatrix_std_lantern = npzread("data/finesst2025/propmatrix_std_lantern.npy")
propmatrix_std_lantern[:,3] += rand(Normal(0, 0.02), 3)
propmatrix_modeselective_lantern = npzread("data/finesst2025/propmatrix_modeselective_lantern.npy")
propmatrix_manufacturing_error_lantern = npzread("data/finesst2025/propmatrix_manufacturing_error_lantern.npy")
propmatrix_manufacturing_error_lantern += rand(Exponential(0.2), (3,3))
propmatrix_manufacturing_error_lantern[1,1] = 0.4

plot(
    im_show(sqrt.(std_lantern_output), aspect_ratio=:equal, c=:binary, colorbar=nothing),
    im_show(sqrt.(modeselective_lantern_output), aspect_ratio=:equal, c=:binary, colorbar=nothing),
    im_show(manufacturing_error_lantern_output, aspect_ratio=:equal, c=:binary, colorbar=nothing),
    im_show(modeselective_lantern_tilt_output, aspect_ratio=:equal, c=:binary, colorbar=nothing),
    heatmap(reverse(propmatrix_std_lantern, dims=1), aspect_ratio=:equal, showaxis=false, legend=nothing, grid=false, c=:matter, clims=(0,1)),
    heatmap(reverse(propmatrix_modeselective_lantern, dims=1), aspect_ratio=:equal, showaxis=false, legend=nothing, grid=false, c=:matter, clims=(0,1)),
    heatmap(reverse(propmatrix_manufacturing_error_lantern, dims=1), aspect_ratio=:equal, showaxis=false, grid=false, legend=false, c=:matter, clims=(0,1)),
    layout=(2,4),
    dpi=1000
)
Plots.savefig("figures/finesst2025_sevenpanel.png")

heatmap(reverse(propmatrix_manufacturing_error_lantern, dims=1), aspect_ratio=:equal, showaxis=false, legend=nothing, grid=false, c=:matter, clims=(0,1))

im_show(sqrt.(std_lantern_output), aspect_ratio=:equal, c=:binary, colorbar=nothing)

A = reverse(propmatrix_manufacturing_error_lantern, dims=1)
heatmap(A, size=(300,300), legend=nothing, c=:matter, axis=nothing, left_margin=-10mm, right_margin=-10mm, top_margin=-10mm, bottom_margin=-10mm)