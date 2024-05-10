using Plots

f_loop = 100 # Hz
f_cutoff = 15 # Hz
α = exp(-2π * f_cutoff / f_loop)

function FH_f(ω)
    α * (1 - exp(-1im * ω)) / (1 - α * exp(-1im * ω))
end

function FL_f(ω)
    (1 - α) * exp(-1im * ω) / (1 - α * exp(-1im * ω))
end

f = 0.0:0.1:50.0
ωr = 2π * (f / f_loop)

FH_vals = abs2.(FH_f.(ωr))
FL_vals = abs2.(FL_f.(ωr))
sum_signal = FH_vals .+ FL_vals

FH_vals, FL_vals = FH_vals ./ sum_signal, FL_vals ./ sum_signal

begin
    p = plot(f, FH_vals, label="High-pass filter", legend=:topright, xlabel="Frequency (Hz)", ylabel="Relative signal strength", dpi=600)
    plot!(f, FL_vals, label="Low-pass filter")
    vline!([f_cutoff], ls=:dash, label="Cutoff frequency", color=4)
    # Plots.savefig("figures/writing/hpf_lpf.png")
    p
end

g = 0.3
s = 1im * ωr
integrator_tf = @. (s) / (g + s)
FHint_f = integrator_tf .* FH_f.(ωr)
FLint_f = integrator_tf .* FL_f.(ωr)

begin
    plot(f, abs.(FHint_f), label="High-pass filter", legend=:right, xlabel="Frequency (Hz)", ylabel="Signal strength")
    plot!(f, abs.(FLint_f), label="Low-pass filter")
    vline!([f_cutoff], ls=:dash, label="Cutoff frequency", color=4)
end