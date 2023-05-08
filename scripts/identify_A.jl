using LinearAlgebra
using PhotonicLantern
using Chain

asincos(sinx, cosx) = sign(sinx) * acos(cosx)

function make_queries(U)
    N = size(U, 1)
     #true_phase_differences = diff(angle.(U), dims=2)
    # true_phase = zeros(N, N)
    # true_phase[:,2:end] = cumsum(true_phase_differences, dims=2)
    amplitude_queries = @chain N ones Diagonal eachrow collect
    amplitude_answers = (x -> abs2.(U * x)).(amplitude_queries)

    cos_phase_queries = [x + y for (x, y) in zip(amplitude_queries[1:end-1], amplitude_queries[2:end])]
    sin_phase_queries = [x + im * y for (x, y) in zip(amplitude_queries[1:end-1], amplitude_queries[2:end])]
    cos_phase_answers = (x -> abs2.(U * x)).(cos_phase_queries)
    sin_phase_answers = (x -> abs2.(U * x)).(sin_phase_queries)
    return amplitude_answers, cos_phase_answers, sin_phase_answers
end

function reconstruct_matrix(amplitude_answers, cos_phase_answers, sin_phase_answers)
    N = length(amplitude_answers)
    # possibly wrong up to a sign on the phase itself
    cos_phase_differences = hcat([
        @. ((p - a1 - a2) / (2 * sqrt(a1) * sqrt(a2)))
        for (p, a1, a2) in zip(cos_phase_answers, amplitude_answers[1:end-1], amplitude_answers[2:end])
    ]...)
    # possibly wrong up to a sign on the overall number, i.e. a pi shift on the phase
    sin_phase_differences = hcat([
        @. ((p - a1 - a2) / (-2 * sqrt(a1) * sqrt(a2)))
        for (p, a1, a2) in zip(sin_phase_answers, amplitude_answers[1:end-1], amplitude_answers[2:end])
    ]...)

    recon_phase_diff = asincos.(sin_phase_differences, cos_phase_differences)
    recon_phase = zeros(N, N)
    recon_phase[:,2:end] = cumsum(recon_phase_diff, dims=2)
    U_recon = Matrix{ComplexF64}(sqrt.(hcat(amplitude_answers...))) .* exp.(1im .* recon_phase)
end

for _ in 1:100
    U = make_unitary_matrix(19)
    q = make_queries(U)
    U_recon = reconstruct_matrix(q...)

    for _ in 1:1000
        xtest = rand(size(U, 1))
        @assert abs2.(U * xtest) ≈ abs2.(U_recon * xtest)
    end
end