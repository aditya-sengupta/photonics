using Plots
pgfplotsx()

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

f_loop = 1000.0
f_cutoff = 50.0
α = exp(-2π * f_cutoff / f_loop)
fr = 0.0:0.1:f_loop/2

function FH(f)
    ω = 2π * f / f_loop
    α * (1 - exp(-1im * ω)) / (1 - α * exp(-1im * ω))
end

function FL(f)
    ω = 2π * f / f_loop
    (1 - α) / (1 - α * exp(-1im * ω))
end

begin
    absFH, absFL = abs2.(FH.(fr)), abs2.(FL.(fr))
    abssum = absFH .+ absFL
    p = plot(fr, absFH ./ abssum, label="High-pass filter response", legend=:right, xlabel="Frequency (Hz)", ylabel="Power (normalized)")
    plot!(fr, absFL ./ abssum, label="Low-pass filter response")
    vline!([f_cutoff], ls=:dash, label="Cutoff frequency")
    Plots.savefig("figures/filter_freqres.pdf")
    p
end