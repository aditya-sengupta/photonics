using NPZ
using LinearAlgebra, Distributions
using Chain
using Plots
using NewtonRaphson
using PhotonicLantern
using Zygote

P = real.(npzread("data/lp_basis.npy"))
N = sqrt(size(P, 2)) |> Int64

begin
    Nmodes = 6
    U = make_unitary_matrix(Nmodes)
    A = U * P
    B, C = interaction_matrices(A)
    screen_lp = 0.1 * rand(MvNormal(zeros(Nmodes), ones(Nmodes))) # phase screen in LP space
    screen_ph = transpose(P) * screen_lp # phase screen in E field space
    test_intensity = abs2.(A * exp.(1im * screen_ph)) # SMF outputs

    quadratic_forward(A, B, C, screen_ph)
    obj = x -> quadratic_forward(A, B, C, x) .- test_intensity
    res = NR(obj, screen_ph, adlib=Zygote)
    sum(abs2, obj(res.x))# .|> abs2 |> sum
end

sum(abs2, obj(screen_ph))

begin
    truth = reshape(screen_ph, (N, N))
    recon = reshape(res.x, (N, N))
    resid = recon .- truth
    r = round(sqrt(sum(abs2, resid)) / N, digits=5)
    p1 = heatmap(truth, title="Original phase screen")
    p2 = heatmap(recon, title="Reconstructed")
    p3 = heatmap(resid, title="Residual: $(r)")
    plot(p1, p2, p3, ticks=nothing, aspect_ratio=:equal)
end