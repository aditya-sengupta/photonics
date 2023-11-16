using FDMModes
using PhotonicLantern
using Plots
using Measures

lambda = 0.635
n_core, n_clad, n_jacket = 1.4592, 1.4504, 1.4496
dx = dy = 0.025
rcore = 1.0
x = y = -(15rcore):dx:(15rcore)

sep1 = 6 * rcore
sep2 = 12 * rcore

thetas_inner = collect(0:(pi/3):(2pi - pi/3))
thetas_outer = collect(0:(pi/6):(2pi - pi/6))

xc = vcat([0.0], sep1 * cos.(thetas_inner), sep2 * cos.(thetas_outer))
yc = vcat([0.0], sep1 * sin.(thetas_inner), sep2 * sin.(thetas_outer))

is_core(x, y) = minimum((x .- xc).^2 + (y .- yc).^2) <= rcore^2
is_clad(x, y) = (x^2 + y^2 <= (196* rcore^2))

index=[is_core(x, y) ? n_core : (is_clad(x, y) ? n_clad : n_jacket) for x=x, y=y]

im_show(index)

neffs, modes = waveguidemodes(index, lambda, dx, dy, nev=20)
scatter(neffs, label="Effective refractive indices") #hline!([n_clad, n_core], label="Guided/radiative cutoff")

plot(im_show.(eachslice(modes, dims=3), clim=(minimum(modes), maximum(modes)), legend=:none, dpi=1000)..., bottom_margin=[-8mm -8mm], left_margin=[-10mm -10mm])
Plots.savefig("figures/livermore_lantern_modes.png")