using PhotonicLantern
using NPZ
using FITSIO
using Base.Meta
using ZernikePolynomials
using Images
using Plots
using Plots.PlotMeasures

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

plus_pl = npzread("data/pl_240122/pl_plus_v2.npy")
minus_pl = npzread("data/pl_240122/pl_minus_v2.npy")
psf_plus = npzread("data/pl_240122/psf_plus_sharp.npy")
psf_plus_plot = im_show(psf_plus[525:555, 1060:1090], c=:grays, colorbar=false, title="Astig.+ science image")
psf_minus = npzread("data/pl_240122/psf_minus_sharp.npy")
psf_minus_plot = im_show(psf_minus[525:555, 1060:1090], c=:grays, colorbar=false, title="Astig.- science image")

function crop(img)
    img[400:900, 600:1100]
end

function minmax(img)
    (img .- minimum(img)) / (maximum(img) - minimum(img))
end

function sinh_ds9(img)
    sinh.(8 * img .^ (1/4))
end

process(img) = sinh_ds9.(minmax(crop(img)))

phase_plus = im_show(evaluateZernike(256, 3, 0.3), colorbar=false, title="Astigmatism+ phase screen")
phase_minus = im_show(evaluateZernike(256, 3, -0.3), colorbar=false, title="Astigmatism- phase screen")

pl_plus = im_show(process(plus_pl), colorbar=false, c=:grays, title="Astig.+ photonic lantern")
pl_minus = im_show(process(minus_pl), colorbar=false, c=:grays, title="Astig.- photonic lantern")
p1 = plot(phase_plus, psf_plus_plot,  pl_plus, phase_minus,psf_minus_plot, pl_minus, titlefontsize=10, dpi=1000, left_margin=-5mm, top_margin=-1mm, bottom_margin=-2mm)
Plots.savefig("figures/norris2020seal.pdf")