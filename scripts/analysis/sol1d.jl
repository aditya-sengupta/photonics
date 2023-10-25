using SparseArrays
using Base: product
using LinearAlgebra: Diagonal, I, diag
using Plots
using Arpack
using Measures

function nanify(x)
    xs = copy(x)
    for idx in findall(abs.(xs) .== 0)
        xs[idx] = NaN
    end
    return xs
end

function im_show(x; kwargs...)
    heatmap(nanify(x), aspect_ratio=1, showaxis=false, grid=false, xlim=(0, size(x, 1) + 2), c=:thermal; kwargs...)
end

function matrix_plot(m, kwargs...)
    heatmap(Matrix(m), yflip=true, aspect_ratio=1, kwargs...)
end

function abs_mode(v)
    N = Int(sqrt(length(v) / 2))
    reshape(abs2.(v[1:N^2]) + abs2.(v[(N^2+1):2N^2]), N, N)
end

function derivative_matrices(N)
    off_axis_x = filter(x->x%N!=0, 1:(N^2-1))
    off_axis_y = 1:(N^2-N)
    x_i = vcat(1:N^2, off_axis_x)
    x_j = vcat(1:N^2, off_axis_x .+ 1)
    x_v = vcat(-1 * ones(N^2), ones(N^2 - N))
    y_i = vcat(1:N^2, off_axis_y)
    y_j = vcat(1:N^2, off_axis_y .+ N)
    y1_v = vcat(-1 * ones(N^2 - 1), ones(N^2 - N + 1))
    y2_v = -copy(y1_v)
    y2_v[N^2] = 1
    Ux = sparse(x_i, x_j, x_v)
    Uy = sparse(y_i, y_j, y1_v)
    Vx = sparse(x_j, x_i, -x_v)
    Vy = sparse(y_j, y_i, y2_v)
    Ux, Uy, Vx, Vy
end

"""
For right now this is just assuming a step-index structure 
and I'm hardcoding in the IOR values
"""
function eps_matrices(N, s, rcore; ncore=1.45, nclad=1.0)
    sr = collect(-s:(2*s/(N-1)):s)
    ior = ones(N, N) * nclad
    core_indices = [(i, j) for (i, j) in product(1:N, 1:N) if sr[i]^2 + sr[j]^2 <= rcore^2]
    for (xi, yi) in core_indices
        ior[xi, yi] = ncore
    end
    eps_r = ior .^ 2
    eps_rx = zeros(N, N)
    eps_rx[2:N, 1:N] .= (view(eps_r, 1:(N-1), 1:N) + (view(eps_r, 2:N, 1:N))) / 2
    eps_rx[1, 1:N] .= view(eps_r, N, 1:N)
    eps_ry = zeros(N, N)
    eps_ry[1:N, 2:N] = (view(eps_r, 1:N, 1:(N-1)) + (view(eps_r, 1:N, 2:N))) / 2
    eps_ry[1:N, 1] = view(eps_r, 1:N, N)
    eps_rz = zeros(N, N)
    eps_rz[2:N, 2:N] = (view(eps_r, 1:(N-1), 1:(N-1)) + view(eps_r, 1:(N-1), 2:N) + view(eps_r, 2:N, 1:(N-1)) + view(eps_r, 2:N, 2:N)) / 4
    eps_rz[1, 1:N] = view(eps_r, 1, 1:N)
    eps_rz[2:N, 1] = view(eps_r, 2:N, 1)
    map(Diagonal ∘ vec, (eps_rx, eps_ry, eps_rz))
end

function wave_matrix(k₀, eps_rx, eps_ry, eps_rz, Ux, Uy, Vx, Vy)
    sqk0 = k₀^2
    invsqk0 = 1 / sqk0
    inv_eps_rz = inv(eps_rz)
    Pxx = -invsqk0 * Ux * inv_eps_rz * Vy * Vx * Uy + (sqk0 * I + Ux * inv_eps_rz * Vx) * (eps_rx + invsqk0 * Vy * Uy)
    Pyy = -invsqk0 * Uy * inv_eps_rz * Vx * Vy * Ux + (sqk0 * I + Uy * inv_eps_rz * Vy) * (eps_ry + invsqk0 * Vx * Ux)
    Pxy = Ux * inv_eps_rz * Vy * (eps_ry + invsqk0 * Vx * Ux) - invsqk0 * (sqk0 * I + Ux * inv_eps_rz * Vx) * Vy * Ux
    Pyx = Uy * inv_eps_rz * Vx * (eps_rx + invsqk0 * Vy * Uy) - invsqk0 * (sqk0 * I + Uy * inv_eps_rz * Vy) * Vx * Uy
    return [Pxx Pxy; Pyx Pyy]
end

N = 201
λ = 1.5 # um
s = 10.0 # um
rcore = 6.0 # um
k₀ = 2pi / λ # 1/um
ds = (2*s)/(N-1) # um
Ux, Uy, Vx, Vy = derivative_matrices(N);
eps_rx, eps_ry, eps_rz = eps_matrices(N, s, rcore);
mask = reshape(diag(eps_rz), N, N)
mask[mask .== minimum(mask)] .= 0.0;
mask[mask .> 0.0] .= 1.0
P = wave_matrix(k₀, eps_rx, eps_ry, eps_rz, Ux, Vx, Uy, Vy) ./ (ds^2);
vals, vecs = eigs(P, nev=30, maxiter=10_000);
eff_ior = abs.(sqrt.(abs.(vals))) * ds / k₀
plot(eff_ior, xlabel="Mode number", ylabel="Effective refractive index", legend=nothing)

modes = [im_show(mask .* abs_mode(vec), colorbar=false) for vec in eachcol(vecs)]
plot(modes[1:30]..., margin=-6.0mm)