# simple case: state [x_pupil, x_NCPA, u], no dynamics - just static correction.
using LinearAlgebra, Distributions, Plots
using StatsBase: mean

rms(x) = sqrt(mean((x .- mean(x)) .^ 2))

A = [1 0 0; 0 1 0; 0 0 0]
B = [0 0 1]'
V = diagm([1e-1, 1e-4, 1e-9]) # less process noise on NCPA

C = [1 0 -1; 1 1 -1]
W = diagm([1e-6, 1e-4])

Q = [1 1 -1; 1 1 -1; -1 -1 1] # minimize (x_pupil + x_NCPA - u)^2

P = diagm(ones(3))
Σ = zeros(3,3)
for _ in 1:Int(1e3)
    P = Q + A' * P * A + V - A' * P * B * inv(B' * P * B) * B' * P * A
    Σ = A * Σ * A' + V - A * Σ * C' * inv(C * Σ * C' + W) * C * Σ * A'
end
L = -inv(B' * P * B) * B' * P * A # LQR gain on the reconstructed state
K = Σ * C' * inv(C * Σ * C' + W) # Kalman gain for observation --> state

x = [0.8, 0.2, 0] # pupil-plane WFS observes 0.5, focal-plane WFS observes 0.5 + 0.1, no DMC
xhat = copy(x)
u = [0]
y = zeros(2)
process_noise = MvNormal(V)
measurement_noise = MvNormal(W)
yv = []
for _ in 1:100
    x = A * x + B * u + rand(process_noise)
    y = C * x + rand(measurement_noise)
    push!(yv, y)
    xhat = A * xhat + B * u - K * y
    u = -L * xhat
end
yv = hcat(yv...)
plot(yv[1,:] .^ 2, xlabel="niter", ylabel="aberration strength", label="pupil-plane, rmse = $(round(rms(yv[1,:]), digits=3))")
plot!(yv[2,:] .^ 2, label="focal-plane, rmse = $(round(rms(yv[2,:]), digits=3))")

