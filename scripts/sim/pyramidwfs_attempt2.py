# %%
import numpy as np
from matplotlib import pyplot as plt
import hcipy as hc
from hcipy import imshow_field
from tqdm import trange

# %%
wavelength_wfs = 842.0E-9
telescope_diameter = 6.5
zero_magnitude_flux = 3.9E10
stellar_magnitude = 0
num_pupil_pixels = 60
pupil_grid_diameter = 60/56 * telescope_diameter
pupil_grid = hc.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
pwfs_grid = hc.make_pupil_grid(120, 2 * pupil_grid_diameter)
magellan_aperture = hc.evaluate_supersampled(hc.make_magellan_aperture(), pupil_grid, 6)
num_actuators_across_pupil = 10
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = hc.make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
deformable_mirror = hc.DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators
pwfs = hc.PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
camera = hc.NoiselessDetector(pwfs_grid)
wf = hc.Wavefront(magellan_aperture, wavelength_wfs)
wf.total_power = 1
camera.integrate(pwfs.forward(wf), 1)
image_ref = camera.read_out()
image_ref /= image_ref.sum()
# Create the interaction matrix
probe_amp = 0.01 * wavelength_wfs
slopes = []

wf = hc.Wavefront(magellan_aperture, wavelength_wfs)
wf.total_power = 1

for ind in trange(num_modes):
    slope = 0

    # Probe the phase response
    for s in [1, -1]:
        amp = np.zeros((num_modes,))
        amp[ind] = s * probe_amp
        deformable_mirror.actuators = amp

        dm_wf = deformable_mirror.forward(wf)
        wfs_wf = pwfs.forward(dm_wf)

        camera.integrate(wfs_wf, 1)
        image = camera.read_out()
        image /= np.sum(image)

        slope += s * (image-image_ref)/(2 * probe_amp)

    slopes.append(slope)

slopes = hc.ModeBasis(slopes)

rcond = 1E-3
reconstruction_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)
# %%
fried_parameter = 0.05  # meter; vary this from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meter (this parameter does not really matter)
velocity = 50.0  # meter/sec
Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2 which our model wants

# make our atmospheric turbulence layer
layer = hc.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
# %%
spatial_resolution = wavelength_wfs / telescope_diameter
focal_grid = hc.make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
prop = hc.FraunhoferPropagator(pupil_grid, focal_grid)
ref_psf = prop(wf)
norm = ref_psf.power.max()
delta_t = 1E-3
leakage = 1.0
gain = 0.8
PSF = prop(deformable_mirror(wf)).power
def pyramid_loop(niter=1, plot_every=0):
    deformable_mirror.random(0.2 * wavelength_wfs)
    camera.integrate(pwfs.forward(deformable_mirror.forward(layer(wf))), 1)
    wfs_image = camera.read_out().astype('float')
    wfs_image /= np.sum(wfs_image)
    psf = prop(deformable_mirror(wf))
    strehls = [float(hc.get_strehl_from_focal(psf.intensity, ref_psf.intensity))]
    for i in range(niter):
        if plot_every > 0 and i % plot_every == 0:
            fig, axs = plt.subplots(1, 2)
            imshow_field(np.log10(psf.intensity) / norm, ax=axs[0])
            imshow_field(wfs_image, ax=axs[1])
            plt.show()
            
        layer.evolve_until(i * delta_t)
        wf_dm = deformable_mirror.forward(layer(wf.copy()))
        wf_pyr = pwfs.forward(wf_dm)

        camera.integrate(wf_pyr, 1)
        wfs_image = camera.read_out().astype('float')
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image - image_ref
        deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)

        phase = magellan_aperture * deformable_mirror.surface
        phase -= np.mean(phase[magellan_aperture>0])
        
        psf = prop(deformable_mirror(wf))
        strehls.append(float(hc.get_strehl_from_focal(psf.intensity, ref_psf.intensity)))
        
    return wfs_image, psf, strehls

# %%
wfs_image, psf, strehls = pyramid_loop(10, 3)

