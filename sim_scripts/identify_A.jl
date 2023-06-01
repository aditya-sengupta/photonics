using LinearAlgebra
using PhotonicLantern
using Chain

# asincos(sinx, cosx) = sign(sinx) * acos(cosx)

function make_queries(U)
    N = size(U, 1)
     #true_phase_differences = diff(angle.(U), dims=2)
    # true_phase = zeros(N, N)
    # true_phase[:,2:end] = cumsum(true_phase_differences, dims=2)
    amplitude_queries = @chain N ones Diagonal eachrow collect
    amplitude_answers = (x -> abs2.(U * x)).(amplitude_queries)

    cos_phase_queries = [x + y for (x, y) in zip(amplitude_queries[1:end-1], amplitude_queries[2:end])]
    sum_phase_queries = [x + y for (x, y) in zip(amplitude_queries[1:end-2], amplitude_queries[3:end])]
    cos_phase_answers = (x -> abs2.(U * x)).(cos_phase_queries)
    sum_phase_answers = (x -> abs2.(U * x)).(sum_phase_queries)
    return amplitude_answers, cos_phase_answers, sum_phase_answers
end

function reconstruct_matrix(amplitude_answers, cos_phase_answers, sum_phase_answers)
    N = length(amplitude_answers)
    # possibly wrong up to a sign on the phase itself
    cos_phase_differences = hcat([
        @. ((p - a1 - a2) / (2 * sqrt(a1) * sqrt(a2)))
        for (p, a1, a2) in zip(cos_phase_answers, amplitude_answers[1:end-1], amplitude_answers[2:end])
    ]...)
    # these are cos(p13 = p12 + p23), cos(p24 = p23 + p34), etc
    sum_cos_phase_differences = hcat([
        @. ((p - a1 - a2) / (2 * sqrt(a1) * sqrt(a2)))
        for (p, a1, a2) in zip(sum_phase_answers, amplitude_answers[1:end-1], amplitude_answers[2:end])
    ]...)

    abs_phase_differences = acos.(cos_phase_differences)
    # cos 12, cos 23, cos 13
    # cos 23 = cos 12 sin 23 + cos 23 sin 12
    # WLOG, sin 12 is +sqrt(1 - cos 12 ^ 2)
    sin_this = @. sqrt(1 - cos_phase_differences[:,1]^2)
    for (i, (cos_this, cos_next, cos_sum)) in enumerate(zip(
        eachcol(cos_phase_differences[:,1:end-1]), 
        eachcol(cos_phase_differences[:,2:end]), 
        eachcol(sum_cos_phase_differences)
    ))
        sin_this = @. (cos_sum - (cos_next * cos_this)) / sin_this
        abs_phase_differences[:,i+1] .*= sign.(sin_this)

    end
    recon_phase = zeros(N,N)
    recon_phase[:,2:end] = cumsum(abs_phase_differences, dims=2)
    U_recon = Matrix{ComplexF64}(sqrt.(hcat(amplitude_answers...))) .* exp.(1im .* recon_phase)
end

for _ in 1:100
    U = make_unitary_matrix(4)
    q = make_queries(U)
    U_recon = reconstruct_matrix(q...)

    for _ in 1:1000
        xtest = rand(size(U, 1))
        @assert abs2.(U * xtest) ≈ abs2.(U_recon * xtest)
    end
end