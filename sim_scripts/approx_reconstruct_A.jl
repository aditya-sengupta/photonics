using LinearAlgebra
using PhotonicLantern

function make_queries(measure, Nmodes=12, Ntest=200)
    basis_inputs = [[i == j ? 1 : 0 for i in 1:Nmodes] for j in 1:Nmodes]
    basis_outputs = map(measure, basis_inputs)

    test_inputs = [[rand() * 2 - 1 for _ in 1:Nmodes] for _ in 1:Ntest]
    test_outputs = map(measure, test_inputs)
    return basis_outputs, test_inputs, test_outputs
end

function reconstruct_row(basis_outputs, test_inputs, test_outputs, Ntest=200)
    Nmodes = length(basis_outputs)
    abs_A = sqrt.(hcat(basis_outputs...))
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
    diff_cosines = reconstruction_matrix \ reconstruction_result
    sine_products = copy(diff_cosines[Nmodes:end])
    n = 1
    for l in 2:Nmodes
        for m in (l+1):Nmodes
            #@assert sine_products[n] ≈ cos(truep_normal[l-1] - truep_normal[m-1])
            sine_products[n] -= diff_cosines[l-1] * diff_cosines[m-1]
            #@assert sine_products[n] ≈ sin(truep_normal[l-1]) * sin(truep_normal[m-1])
            n += 1
        end
    end
    # @assert maximum(sine_products) <= 1 && minimum(sine_products) >= -1
    signs = vcat(1, sign.(sine_products[1:Nmodes-2]))

    recon_phases = vcat(0, acos.(diff_cosines[1:Nmodes-1]) .* signs)
    return abs_A .* exp.(1im * recon_phases)'
end

function test_matrix_recon(Nports=18, Nmodes=12)
    A = make_unitary_matrix(Nports)[:,1:Nmodes]
    A_recon = zeros(ComplexF64, size(A))
    for r in 1:Nports
        A_recon[r:r,:] = reconstruct_row(
            make_queries(x -> abs2.(A[r:r,:] * x))...
        )
    end
    for _ in 1:100
        x = rand(Nmodes)
        @assert abs2.(A * x) ≈ abs2.(A_recon * x)
    end
end

test_matrix_recon()

reconstruct_row(make_queries(x -> abs2.(A[1:1,:] * x))...)

for r in 1:Nmodes
    A = A[r:r,:]
    test_inputs = [[rand() * 2 - 1 for _ in 1:Nmodes] for _ in 1:Ntest]
    test_outputs = map(x -> abs2.(A * x), test_inputs)

    basis_inputs = [[i == j ? 1 : 0 for i in 1:Nmodes] for j in 1:Nmodes]
    basis_outputs = map(x -> abs2.(A * x), basis_inputs)
    abs_A = sqrt.(hcat(basis_outputs...))
    # @assert abs_A ≈ abs.(A)

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

    true_phases = vec(angle.(A))
    diff_cosines = reconstruction_matrix \ reconstruction_result
    n = 1
    for l in 1:Nmodes
        for m in (l+1):Nmodes
            @assert diff_cosines[n] ≈ cos(true_phases[l] - true_phases[m])
            n += 1
        end
    end

    #truep_normal = true_phases[2:end] .- true_phases[1]
    #@assert cos.(truep_normal) ≈ diff_cosines[1:Nmodes-1]
    #@assert truep_normal ≈ acos.(diff_cosines[1:Nmodes-1]) .* sign.(truep_normal)

    sine_products = copy(diff_cosines[Nmodes:end])
    n = 1
    for l in 2:Nmodes
        for m in (l+1):Nmodes
            #@assert sine_products[n] ≈ cos(truep_normal[l-1] - truep_normal[m-1])
            sine_products[n] -= diff_cosines[l-1] * diff_cosines[m-1]
            #@assert sine_products[n] ≈ sin(truep_normal[l-1]) * sin(truep_normal[m-1])
            n += 1
        end
    end
    @assert maximum(sine_products) <= 1 && minimum(sine_products) >= -1
    signs = vcat(1, sign.(sine_products[1:Nmodes-2]))

    recon_phases = vcat(0, acos.(diff_cosines[1:Nmodes-1]) .* signs)
    recon_A = abs_A .* exp.(1im * recon_phases)'
    for _ in 1:100
        x = rand(Nmodes)
        @assert abs2.(A * x) ≈ abs2.(recon_A * x)
    end
end

