# %% imports
import numpy as np
import hcipy
from tqdm import trange

from matplotlib import pyplot as plt
from matplotlib import animation
from hcipy import imshow_field
from IPython.display import HTML

from photonics import optical_setup, dm_setup, bin_image, pyramid_slopes

# %% Setting parameters
ao_config = {
    # Optical parameters
    "grid_size" : 120, # pixels
    "D" : 1, # meters
    "wavelength" : 750e-9, # meters
    # Turbulence parameters
    "Cn_squared" : hcipy.Cn_squared_from_fried_parameter(0.05), # the parameter is the Fried parameter in meters; vary this from 0.05 (really bad) to 0.2 (really good)
    "outer_scale" : 50, # meter (this parameter does not really matter)
    "vx" : 10.0,
    "vy" : 0.0, # windspeed in m/s
    # DM parameters
    "num_actuators" : 9, # for the DM
    # WFS parameters
    # Controller parameters
    "gain" : 0.0,
    "leak" : 0.999,
    "dt" : 1.0 / 80.0
}
# %% Optical setup
aperture, pupil_grid, focal_grid, propagator, wf_init = optical_setup(ao_config)
# %% Atmospheric turbulence layer
layer = hcipy.InfiniteAtmosphericLayer(
    pupil_grid, 
    ao_config["Cn_squared"], 
    ao_config["outer_scale"], 
    np.sqrt(ao_config["vx"]**2 + ao_config["vy"]**2)
)
# %% Deformable mirror
deformable_mirror = dm_setup(pupil_grid, ao_config)
num_modes = deformable_mirror.num_actuators
deformable_mirror.random(0.2 * ao_config["wavelength"])

# %%
# setup Pyramid WFS
pwfs_grid = hcipy.make_pupil_grid(120, 120/56)
pwfs = hcipy.PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=60/56, pupil_diameter=ao_config["D"], wavelength_0=ao_config["wavelength"], q=3)
wfs_camera = hcipy.NoiselessDetector(pwfs_grid)
wfs_camera.integrate(pwfs.forward(wf_init), 1)
image_ref = wfs_camera.read_out()
image_ref /= image_ref.sum()

# %% Command matrix
# This code will have to be rerun everytime you change a parameter about the PyWFS or DM.
# Create the interaction matrix
probe_amp = 0.01 * ao_config["wavelength"]
slopes = []

wf = hcipy.Wavefront(aperture, ao_config["wavelength"])
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

        wfs_camera.integrate(wfs_wf, 1)
        image = wfs_camera.read_out()
        image /= np.sum(image)

        slope += s * (image-image_ref)/(2 * probe_amp)

    slopes.append(slope)

slopes = hcipy.ModeBasis(slopes)
# %%
rcond = 1E-3
CM = hcipy.inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)
# %%
spatial_resolution = ao_config["wavelength"] / ao_config["D"]
focal_grid = hcipy.make_focal_grid(q=8, num_airy=20, spatial_resolution=spatial_resolution)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
norm = prop(wf).power.max()
# %%
def create_closed_loop_animation():
    
    PSF = prop(deformable_mirror(wf)).power
    
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
    
    def animate(t):
        wf_in = wf.copy()
        layer.evolve_until(t)
        wf_dm = deformable_mirror.forward(layer.forward(wf_in))
        wf_pyr = pwfs.forward(wf_dm)

        wfs_camera.integrate(wf_pyr, 1)
        wfs_image = wfs_camera.read_out().astype('float')
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image - image_ref
        deformable_mirror.actuators = ao_config["leak"] * deformable_mirror.actuators - ao_config["gain"] * CM.dot(diff_image)

        phase = aperture * deformable_mirror.surface
        phase -= np.mean(phase[aperture>0])
        
        psf = prop(deformable_mirror(wf) ).power
        
        im1.set_data(*pupil_grid.separated_coords, (aperture * deformable_mirror.surface).shaped / 1e-6)
        im2.set_data(*pwfs_grid.separated_coords, wfs_image.shaped)
        im3.set_data(*focal_grid.separated_coords, np.log10(psf.shaped / norm))

        return [im1, im2, im3]
    
    num_time_steps=21
    time_steps = np.arange(num_time_steps)
    anim = animation.FuncAnimation(fig, animate, time_steps, interval=160, blit=True)
    return HTML(anim.to_jshtml(default_mode='loop'))
    
create_closed_loop_animation()
# %%
