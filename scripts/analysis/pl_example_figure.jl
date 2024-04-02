using ZernikePolynomials
using PhotonicLantern
using FFTW
using PythonCall

hcipy = pyimport("hcipy")
numpy = pyimport("numpy")
function aberration_to_phase_psf(zern, amp, diam=1)
    pupil_grid = hcipy.make_pupil_grid(256)
    telescope_pupil_generator = hcipy.make_circular_aperture(diameter=diam)
    telescope_pupil = telescope_pupil_generator(pupil_grid)
    q = 1im * amp * hcipy.mode_basis.zernike_ansi(zern, diam)(pupil_grid)
    phase = amp * hcipy.mode_basis.zernike_ansi(zern, diam)(pupil_grid)
    wavefront = hcipy.Wavefront(numpy.exp(q))
    focal_grid = hcipy.make_focal_grid(q=200, num_airy=3)
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
    focal_image = prop.forward(wavefront)
    img = (focal_image.intensity / focal_image.intensity.max()).shaped |> PyArray |> Matrix
    sqrt.(img), phase.shaped |> PyArray |> Matrix
end

im_show(aberration_to_phase_psf(5, -0.5)[1])