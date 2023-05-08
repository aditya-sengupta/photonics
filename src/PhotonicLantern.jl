module PhotonicLantern
    using Tullio
    using SpecialFunctions
    ENV["JULIA_CONDAPKG_BACKEND"] = :Null
    ENV["JULIA_PYTHONCALL_EXE"] = "./.venv/bin/python"
    using PythonCall
    using LinearAlgebra: I
    # lightbeam = pyimport("lightbeam")

    """
    Makes the ideal interaction matrices B and C out of a forward matrix A.
    """
    function interaction_matrices(A)
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

    export interaction_matrices, quadratic_forward, quadratic_jacobian, lpfield, make_unitary_matrix, is_unitary
end # module PhotonicLantern
