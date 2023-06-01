using PhotonicLantern
using NewtonRaphson
using LinearAlgebra
using Plots
using ProgressMeter
using InvertedIndices
using BenchmarkTools

rms(x) = sqrt(sum(x .^ 2))
n_in, n_out = 10, 6

function matrices_for_condition_number(condition)
    A = rand(ComplexF64, n_out, n_in);
    F = svd(A)
    singulars = F.S
    log_min_s = log(singulars[end])
    step = log(condition) / (min(n_in, n_out) - 1)
    singulars = exp.(log_min_s:step:(log_min_s + log(condition))) |> collect
    A = F.U * Diagonal(reverse(singulars)) * F.Vt
    B, C = interaction_matrices(A)
    return A, B, C
end

begin
    condition_numbers = []
    phase_resids = []
    successes = []
    @showprogress for target_condition_number in [2, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 80000, 1e6, 5e6, 1e7, 1e8]
        A, B, C = matrices_for_condition_number(target_condition_number)
        for _ in 1:10
            dϕ = 0.1 * rand(n_in)
            p_true = quadratic_forward(A, B, C, dϕ)
            obj(x) = quadratic_forward(A, B, C, x) - p_true
            dϕ_guess = 0.1 * rand(n_in)
            # println("Initial residual: $(rms(obj(dϕ_guess) - p_true))")
            res = NR(obj, dϕ_guess)
            push!(phase_resids, rms(dϕ - res.x))
            push!(condition_numbers, target_condition_number)
            push!(successes, res.retcode == :success && res.iters > 0)
            # println("Final residual: $(rms(obj(res.x)))")
            # println("Phase residual: $(rms(dϕ - res.x))")
        end
    end
end

begin
    success_mask = findall(successes)
    n_s, n_f = length(success_mask), length(successes) - length(success_mask)
    plot(xlabel="Condition number of propagation matrix", ylabel="Residual in phase inversion solution", legend=:bottomright)
    scatter!(condition_numbers[success_mask], phase_resids[success_mask], xscale=:log10, yscale=:log10, label="Success ($n_s)", ms=2, msw=0, color=1)
    scatter!(condition_numbers[Not(success_mask)], phase_resids[Not(success_mask)], label="Failure ($n_f)", ms=2, msw=0, color=:red)
end
