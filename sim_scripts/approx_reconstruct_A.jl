using LinearAlgebra
using PhotonicLantern
using NewtonRaphson
using ForwardDiff

asincos(sinx, cosx) = sign(sinx) * acos(cosx)

Ntest = 200
Nmodes = 12
A_all = make_unitary_matrix(18)[:,1:Nmodes]
r = 1
x = rand(Nmodes)
begin
    A = A_all[r:r,:]
    measure(x) = abs2.(A * x)
    test_inputs = [[rand() * 2 - 1 for _ in 1:Nmodes] for _ in 1:Ntest]
    test_outputs = map(measure, test_inputs)

    basis_inputs = [[i == j ? 1 : 0 for i in 1:Nmodes] for j in 1:Nmodes]
    basis_outputs = map(measure, basis_inputs)
    abs_A = sqrt.(hcat(basis_outputs...))
    @assert abs_A ≈ abs.(A)

    reconstruction_matrix = zeros(Ntest, Nmodes * (Nmodes - 1) ÷ 2)
    reconstruction_result = zeros(Ntest)

    for (j, (inp, outp)) in enumerate(zip(test_inputs, test_outputs))
        coeff_magnitudes = hcat([abs_A[:,k] * inp[k] for k in 1:Nmodes]...)
        reconstruction_result[j,:] = outp .- sum(abs2, coeff_magnitudes)
        # go in the order (1, 2), (1, 3), ... (1, Nmodes), (2, 3), ... (2, Nmodes), ..., (Nmodes - 1, Nmodes)
        n = 1
        for l in 1:Nmodes
            for m in (l+1):Nmodes
                reconstruction_matrix[j, n] = 2 * coeff_magnitudes[l] .* coeff_magnitudes[m]
                n += 1
            end
        end
    end

    angA = angle.(A)
    true_phases = vec(angA)
    diff_cosines = reconstruction_matrix \ reconstruction_result
    true_diff_cosines = zeros(length(diff_cosines))
    n = 1
    for l in 1:Nmodes
        for m in (l+1):Nmodes
            true_diff_cosines[n] = cos(angA[l] - angA[m])
            n += 1
        end
    end
    @assert diff_cosines ≈ true_diff_cosines

   
end

function obj(sincosphases)
    target_cosines = zeros(n - 1)
    k = 1
    for l in 1:Nmodes
        for m in (l+1):Nmodes
            target_cosines[k] = ForwardDiff.value(sincosphases[l] * sincosphases[m] + sincosphases[Nmodes+l] * sincosphases[Nmodes+m])
            k += 1
        end
    end
    return target_cosines .- diff_cosines
end

