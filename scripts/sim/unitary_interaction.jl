using PhotonicLantern
using Base: product

N = 6
A = make_unitary_matrix(N)
B, _ = interaction_matrices(A)
B

amps = collect(-1.5:0.1:1.5)

all_intensities = zeros(N, length(amps), N)
for (i, j) in product(1:N, 1:length(amps))
    inp = zeros(N)
    inp[i] = amps[j]
    all_intensities[i, j, :] .= abs2.(A * inp)
end

all_intensities[:,17,:] * inv(B)