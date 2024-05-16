using NPZ
using Images
using ZernikePolynomials
using PhotonicLantern
using Plots
using Plots.PlotMeasures
using LaTeXStrings

rms(x) = sqrt(sum(x .^ 2))

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

λ = 635.0 # nm

rms_before = npzread("./data/pl_231023/before_monitoring.npy")
amps_during = npzread("./data/pl_231023/cl_recons.npy")
rms_during = [rms(r) for r in eachrow(amps_during)]
rms_after = npzread("./data/pl_231023/after_monitoring.npy")
amp_init = npzread("./data/pl_231023/success_amp.npy")
amp_final = npzread("./data/pl_231023/final_slm.npy")

phase_screen_init = evaluateZernike(256, collect(1:11), amp_init)
phase_screen_final = evaluateZernike(256, collect(1:11), amp_final)

psf_before = Float64.(Gray.(load("./data/pl_231025/psf_aberrated.png")))
psf_after = Float64.(Gray.(load("./data/pl_231025/psf_corrected.png")))
m = 490:590,1020:1120
heatmap(log10.(psf_before)[m...], c=:grays)
heatmap(log10.(psf_after)[m...], c=:grays)

begin
    p = plot(vcat(rms_before, rms_during, rms_after) / (2pi / λ), xlabel="Number of iterations", ylabel="Wavefront error measured on the PL (nm)", label=nothing, ylim=(0, 170), title="Correcting a static aberration with the photonic lantern", titlefontsize=13, dpi=800)
    vspan!([20, 60], color=RGBA(0.2, 1, 0.2, 1), alpha=0.3, label="Loop closed")
    BB1 = bbox(0.01, 0.2, 0.3, 0.3)
    BB2 = bbox(0.65, 0.2, 0.3, 0.3)
    BB3 = bbox(0.01, 0.6, 0.2, 0.3)
    BB4 = bbox(0.65, 0.6, 0.2, 0.3)
    im_show!(phase_screen_init * (λ / (2pi)), inset=(1, BB1), clim=(-λ/2, λ/2), subplot=2, title="Initial phase (nm)", titlefontsize=7, ytickfontsize=6, background_alpha_subplot=0.0, background_color_subplot=RGBA(0, 0, 0, 0), topmargin=-2mm)
    im_show!(phase_screen_final * (λ / (2pi)), inset=(1, BB2), clim=(-λ/2, λ/2), subplot=3, title="Final phase (nm)", titlefontsize=7, background_color_subplot=RGBA(0, 0, 0, 0), background_alpha_subplot=0.0, ytickfontsize=6, topmargin=-2mm)
    heatmap!(log10.(psf_before[m...]), c=:grays, showaxis=false, grid=false, inset=(1, BB3), subplot=4, title="Initial image", titlefontsize=7, xlim=(0, length(m[1])), ylim=(0, length(m[2])), legend=nothing, yflip=true, topmargin=-2mm)
    heatmap!(log10.(psf_after[m...]), c=:grays, showaxis=false, grid=false, inset=(1, BB4), subplot=5, title="Final image", titlefontsize=7, xlim=(0, length(m[1])), ylim=(0, length(m[2])), legend=nothing, yflip=true, topmargin=-2mm)
    Plots.savefig("./figures/seal_static_cl.pdf")
    p
end
