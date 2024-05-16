module PhotonicLantern
    using Tullio
    using SpecialFunctions
    using LinearAlgebra: I
    using Plots

    """
    Makes the ideal interaction matrices B and C out of a forward matrix A.
    """
    function interaction_from_forward(A)
        cA = conj(A)
        @tullio interaction[i,j] := cA[i,j] * A[i,k]
        return 2 * imag(interaction), 2 * real(interaction)
    end

    function quadratic_forward(A, B, C, dϕ)
        A_term1 = abs2.(A * dϕ)
        A_term2 = abs2.(sum(A, dims=2))
        B_term = B * dϕ
        C_term = -(1/2) * C * (dϕ .^ 2)
        return A_term1 + A_term2 + B_term + C_term
    end

    function quadratic_jacobian(A, B, C, dϕ)
        A2mC = abs2.(A) .- C
        @tullio jac2[i,j] := A[i,j] * conj(A[i,k]) * dϕ[k]
        return B .- A2mC * dϕ + jac2
    end

    function get_b(l, m, V)
        return pyconvert(Float64, lightbeam.get_b(l, m, V))
    end

    function lpfield(x, y, l, m, V, a, azimuthal=cos) 
        b = 0.9 # pyconvert(Float64, lightbeam.get_b(l, m, V))
        u, v = V * sqrt(1 - b), V * sqrt(b)
        az = azimuthal(l * atan(y, x))
        r = sqrt(x^2 + y^2)
        if r <= a
            return az * besselj(l, r / a)
        else
            return az * besselj(l, u) / besselk(l, v) * besselk(l, v * r / a)
        end
    end

    function make_unitary_matrix(N)
        H = 2 .* rand(ComplexF64, N, N) .- (1 + 1im)
        exp(1im * (H + H'))
    end
    
    function is_unitary(M)
        M * M' ≈ I && M' * M ≈ I
    end

    sigma_clamp(x) = clamp(σ(x), 0.15, 0.85)
    zero_one_ify(x) = (x .- minimum(x, dims=2)) ./ (maximum(x, dims=2) .- minimum(x, dims=2)), minimum(x, dims=2)[:,1], maximum(x, dims=2)[:,1]

    function nanify(x)
        xs = copy(x)
        for idx in findall(abs.(xs) .== 0)
            xs[idx] = NaN
        end
        return xs
    end
    
    function im_show(x; kwargs...)
        heatmap(nanify(x), aspect_ratio=1, showaxis=false, grid=false, xlim=(0, size(x, 1) + 2), c=:RdBu; kwargs...)
    end

    function im_show!(x; kwargs...)
        heatmap!(nanify(x), aspect_ratio=1, showaxis=false, grid=false, xlim=(0, size(x, 1) + 2), c=:RdBu; kwargs...)
    end

    function phasewrap(x)
        return mod(x + π, 2 * π) - π
    end
    
    function rescale(v, vmin::Vector, vmax::Vector)
        return v .* (vmax .- vmin) .+ vmin
    end

    include("interaction.jl")

    export interaction_matrices, quadratic_forward, quadratic_jacobian, lpfield, make_unitary_matrix, is_unitary, sigma_clamp, zero_one_ify, im_show, im_show!, phasewrap, rescale
end # module PhotonicLantern
