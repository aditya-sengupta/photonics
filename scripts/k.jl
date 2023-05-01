using LinearAlgebra
using ForwardDiff: jacobian
using Zygote

A = rand(ComplexF64, 5, 5)
rA, iA = real.(A), imag.(A)
B = rand(5, 5)
C = rand(5, 5)
ϕ = rand(5)

g(x) = B * x - (1/2) * C * x .^ 2; all(jacobian(g, ϕ) .≈ B - C * diagm(ϕ))

f(x) = abs2.(A * x)
jacobian(f, ϕ)
jacobian(x -> (rA * x) .^ 2 + (iA * x) .^ 2, ϕ)

jacobian(x -> (rA * x) .^ 2, ϕ)
