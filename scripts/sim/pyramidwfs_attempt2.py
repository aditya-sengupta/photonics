# %%
import numpy as np
from matplotlib import pyplot as plt
import hcipy as hc
from hcipy import imshow_field
from tqdm import trange, tqdm
from copy import copy
from photonics.lantern_optics import LanternOptics
from scipy.interpolate import griddata
from matplotlib import animation
from IPython.display import HTML
# %%
lo = LanternOptics(f_number=7)
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
num_actuators_across_pupil = 20
actuator_spacing = telescope_diameter / num_actuators_across_pupil
influence_functions = hc.make_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing)
deformable_mirror = hc.DeformableMirror(influence_functions)
num_modes = deformable_mirror.num_actuators
pwfs = hc.PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=pupil_grid_diameter, pupil_diameter=telescope_diameter, wavelength_0=wavelength_wfs, q=3)
# set up pyramid wavefront sensor
pixels_pyramid_pupils=12 # number of pixels across each pupil; want 120 %(mod) pixels_pyramid_pupils =0. VARY THIS PARAMETER
pwfs_grid = hc.make_pupil_grid(pixels_pyramid_pupils*2, telescope_diameter*2)

mld=5 # modulation radius in lambda/D (we will discuss modulation in a future lecture. Do not vary.)
modradius = mld*wavelength_wfs/telescope_diameter # modulation radius in radians;
modsteps = 12 # keep this as a factor of 4. Increasing this value significantly increases the computation time. No need to change today.

pwfs = hc.PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=telescope_diameter, wavelength_0=wavelength_wfs)
mpwfs = hc.ModulatedPyramidWavefrontSensorOptics(pwfs,modradius,modsteps)
wfs_camera = hc.NoiselessDetector(pwfs_grid)
wf = hc.Wavefront(magellan_aperture, wavelength_wfs)
wf.total_power = 1
wfs_camera.integrate(pwfs.forward(wf), 1)
image_ref = wfs_camera.read_out()
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

        for m in range (modsteps) :
            wfs_camera.integrate(mpwfs(dm_wf)[m], 1)
        image = wfs_camera.read_out()
        image /= np.sum(image)

        slope += s * (image-image_ref)/(2 * probe_amp)

    slopes.append(slope)

slopes = hc.ModeBasis(slopes)

rcond = 1E-3
reconstruction_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=rcond, svd=None)
# %%
fried_parameter = 0.3  # meter; vary this from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meter (this parameter does not really matter)
velocity = 1.0  # meter/sec
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
leakage = 0.01
gain = 0.4
PSF = prop(deformable_mirror(wf)).power
def pyramid_loop(niter=1, plot_every=0):
    # deformable_mirror.random(0.2 * wavelength_wfs)
    wfs_camera.integrate(pwfs.forward(deformable_mirror.forward(layer(wf))), 1)
    wfs_image = wfs_camera.read_out().astype('float')
    wfs_image /= np.sum(wfs_image)
    psf = prop(deformable_mirror(wf))
    strehls = [float(hc.get_strehl_from_focal(psf.intensity, ref_psf.intensity))]
    focal_images = []
    for i in trange(niter):
        if plot_every > 0 and i % plot_every == 0:
            fig, axs = plt.subplots(1, 2)
            imshow_field(np.log10(psf.intensity) / norm, ax=axs[0])
            imshow_field(wfs_image, ax=axs[1])
            plt.show()
            
        layer.evolve_until(i * delta_t)
        wf_dm = deformable_mirror.forward(layer(wf.copy()))
        wf_pyr = mpwfs.forward(wf_dm)
        for mmm in range (modsteps) :
              wfs_camera.integrate(wf_pyr[mmm], delta_t/modsteps)
        wfs_image = wfs_camera.read_out().astype('float')
        wfs_image /= np.sum(wfs_image)

        diff_image = wfs_image - image_ref
        deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * reconstruction_matrix.dot(diff_image)

        phase = magellan_aperture * deformable_mirror.surface
        
        phase -= np.mean(phase[magellan_aperture>0])
        
        psf = prop(wf_dm)
        focal_images.append(copy(psf))
        strehls.append(float(hc.get_strehl_from_focal(psf.intensity, ref_psf.intensity)))
        
    return focal_images, strehls

# %%
focal_images, strehls = pyramid_loop(50, 0)
# %%
lo.load_outputs()
scaling_factor = np.max(focal_images[0].grid.x) / (lo.f_number * lo.wl)
_, _, lantern_ref = lo.lantern_output(lo.zernike_to_focal(1, 0.0))
# %%
def focal_to_lantern(img):
    d = griddata((img.grid.x / scaling_factor, img.grid.y / scaling_factor), img.electric_field, (lo.focal_grid.x, lo.focal_grid.y), fill_value=0.0+0.0j)
    wf_lantern_input = hc.Wavefront(hc.Field(d, lo.focal_grid), wavelength=lo.wl)
    return lo.lantern_output(wf_lantern_input)
# %%
lantern_images = []
for img in tqdm(focal_images):
    _, _, lantern_img = focal_to_lantern(img)
    lantern_images.append(lantern_img)
    
# %%
def normalize(x):
    return x / np.max(x)
# %%
def create_closed_loop_animation():
    fig = plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    plt.title('PSF input to lantern')
    im1 = imshow_field(np.log10(focal_images[0].intensity / norm))
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title('Second-stage lantern readings')
    im2 = plt.imshow(normalize(np.abs(lantern_images[0]) ** 2))
    plt.colorbar()

    plt.close(fig)

    def animate(t):
        im1.set_data(*focal_images[t].electric_field.grid.separated_coords, np.log10(focal_images[t].intensity / norm).shaped)
        im2.set_data(normalize(np.abs(lantern_images[t]) ** 2))

        return [im1, im2]

    num_time_steps = len(focal_images)
    time_steps = np.arange(num_time_steps)
    anim = animation.FuncAnimation(fig, animate, time_steps, interval=160, blit=True)
    return HTML(anim.to_jshtml(default_mode='loop'))

anim = create_closed_loop_animation()
# %%
