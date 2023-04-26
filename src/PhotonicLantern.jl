module PhotonicLantern
    using Tullio
    """
    Makes the ideal interaction matrices B and C out of a forward matrix A.
    """
    function interaction_matrices(A)
        cA = conj(A)
        @tullio interaction[i,j] := cA[i,j] * A[i,k]
        return 2 * imag(interaction), 2 * real(interaction)
    end

    function quadratic_forward(A, B, C, dϕ)
        @tullio A_term[i] := abs2(A[i,j]^2 + A[i,j] * dϕ[j])
        B_term = B * dϕ
        C_term = -(1/2) * C * (dϕ .* dϕ)
        return A_term + B_term + C_term
    end

    function quadratic_jacobian(A, B, C, dϕ)
        @tullio jac[i,j] := B[i,j] + (abs2(A[i,j]) - C[i,j]) * dϕ[j] + A[i,j] * conj(A[i,k]) * dϕ[k]
        return jac
    end

    export interaction_matrices, quadratic_forward, quadratic_jacobian
end # module PhotonicLantern
