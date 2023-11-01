# %%
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from matplotlib import animation, rc
from IPython.display import HTML

wavelength_wfs = 842.0E-9
telescope_diameter = 6.5

num_pupil_pixels = 60
pupil_grid_diameter = 60/56 * telescope_diameter
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

pwfs_grid = make_pupil_grid(120, 2 * pupil_grid_diameter)

aperture = evaluate_supersampled(make_circular_aperture(telescope_diameter), pupil_grid, 6)

num_actuators_across_pupil = 12
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators

pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
camera = NoiselessDetector(pwfs_grid)

wf = Wavefront(aperture, wavelength_wfs)
wf.total_power = 1

camera.integrate(pwfs.forward(wf), 1)

image_ref = camera.read_out()
image_ref /= image_ref.sum()

# Create the interaction matrix
probe_amp = 0.01 * wavelength_wfs
slopes = []

wf = Wavefront(aperture, wavelength_wfs)
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

slopes = ModeBasis(slopes)

rcond = 1E-3
CM = inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)

spatial_resolution = wavelength_wfs / telescope_diameter
focal_grid = make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
norm = np.max(prop(wf).intensity)
# %%
# let's pick the properties of our turblence
fried_parameter = 0.2  # meter; vary this from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meter (this parameter does not really matter)
vx,vy=10.,0. # windspeed in m/s   vary this
velocity = np.sqrt(vx**2.+vy**2.)  # meter/sec
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2 which our model wants

# make our atmospheric turbulence layer
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

 # %%
delta_t = 1/800
leakage = 0.999
gain = 0.3

PSF = prop(deformable_mirror(layer.forward(wf))).power

fig = plt.figure(figsize=(14,3))
plt.subplot(1,3,1)
plt.title(r'DM surface shape ($\mathrm{\mu}$m)')
im1 = imshow_field(deformable_mirror.surface/(1e-6), vmin=-1, vmax=1, cmap='bwr')
plt.colorbar()    

plt.subplot(1,3,2)
plt.title('Wavefront sensor output')
im2 = imshow_field(image_ref, pwfs_grid)
plt.colorbar()

plt.subplot(1,3,3)
plt.title('Science image plane')
im3 = imshow_field(np.log10(PSF / norm), vmax=0, vmin=-5, cmap='inferno')
plt.colorbar()

plt.close(fig)

num_timesteps = 100

for timestep in trange(num_timesteps):
    wf_in = wf.copy()
    layer.t = timestep * delta_t
    wf_dm = deformable_mirror.forward(layer(wf_in))
    wf_pyr = pwfs.forward(wf_dm)

    camera.integrate(wf_pyr, 1)
    wfs_image = camera.read_out().astype('float')
    wfs_image /= np.sum(wfs_image)

    diff_image = wfs_image - image_ref
    deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * CM.dot(diff_image)

    phase = aperture * deformable_mirror.surface
    phase -= np.mean(phase[aperture>0])
    
    wf_focal = prop(wf_dm)
    
im1.set_data(*pupil_grid.separated_coords, (aperture * deformable_mirror.surface).shaped / 1e-6)
im2.set_data(*pwfs_grid.separated_coords, wfs_image.shaped)
im3.set_data(*focal_grid.separated_coords, np.log10(wf_focal.intensity.shaped / norm))
fig

# %%
