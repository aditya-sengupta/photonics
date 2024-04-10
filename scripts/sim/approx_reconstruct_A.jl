using NPZ
using PhotonicLantern
using ZernikePolynomials
using Plots

im_show(x; kwargs...) = heatmap(x, aspect_ratio=1, showaxis=false, grid=false, xlim=(0, size(x, 1) + 2); kwargs...)

function make_queries(measure, Nmodes, Ntest=200)
    basis_inputs = [[i == j ? 1 : 0 for i in 1:Nmodes] for j in 1:Nmodes]
    basis_outputs = map(measure, basis_inputs)

    test_inputs = [[rand() * 2 - 1 for _ in 1:Nmodes] for _ in 1:Ntest]
    test_outputs = map(measure, test_inputs)
    return basis_outputs, test_inputs, test_outputs
end

function reconstruct_row(basis_outputs, test_inputs, test_outputs)
    Nmodes = length(basis_outputs)
    Ntest = length(test_outputs)
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

function test_matrix_recon(;Nports=18, Nmodes=5)
    A = make_unitary_matrix(Nports)[:,1:Nmodes]
    A_recon = zeros(ComplexF64, size(A))
    for r in 1:Nports
        A_recon[r:r,:] = reconstruct_row(
            make_queries(x -> abs2.(A[r:r,:] * x), Nmodes)...
        )
    end
    for _ in 1:100
        x = rand(Nmodes)
        @assert abs2.(A * x) ≈ abs2.(A_recon * x)
    end
end

test_matrix_recon()

inputzs = npzread("./data/pl_230530/inputzs_230530_1504.npy")
intensities = npzread("./data/pl_230530/pl_intensities_230530_1514.npy")

begin
    N_eval = 256
    mask = evaluateZernike(N_eval, [0], [1.0], index=:OSA)
    electric_field_basis = hcat([vcat(mask .* exp.(1im * evaluateZernike(N_eval, [b], [0.1], index=:OSA))...) for b in 2:19]...)

    random_field = mask .* exp.(1im * evaluateZernike(N_eval, collect(2:1+size(inputzs, 2)), inputzs[903,:], index=:OSA))
    p1 = im_show(angle.(random_field), title="Random phase screen")

    random_field_coeffs = (electric_field_basis \ vcat(random_field...))
    random_field_proj = reshape(electric_field_basis * random_field_coeffs, (N_eval, N_eval))

    p2 = im_show(angle.(random_field_proj), title="Phase in basis")
    p3 = im_show((angle.(random_field) .- angle.(random_field_proj)) .% π, title="Residual phase")
    plot(p1, p2, p3)
end