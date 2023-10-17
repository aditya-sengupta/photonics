# %%
from hcipy import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
# %%
def make_command_matrix(deformable_mirror, pwfs, wfs_camera, wf):

  probe_amp = 0.02 * wf.wavelength
  response_matrix = []
  num_modes=deformable_mirror.num_actuators

  for i in trange(int(num_modes)):
      slope = 0

      for s in [1, -1]:
          amp = np.zeros((num_modes,))
          amp[i] = s * probe_amp
          deformable_mirror.flatten()
          deformable_mirror.actuators = amp

          dm_wf = deformable_mirror.forward(wf)
          wfs_wf = pwfs.forward(dm_wf)

          wfs_camera.integrate(wfs_wf, 1)

          image_nophot = wfs_camera.read_out()
          image_nophot/=image_nophot.sum()

          slope += s * (image_nophot-img_ref)/(2*probe_amp)  #these are not really slopes; this is just a normalized differential image

      response_matrix.append(slope.ravel())

  response_mtx= ModeBasis(response_matrix)
  rcond = 1e-3

  reconstruction_matrix = inverse_tikhonov(response_mtx.transformation_matrix, rcond=rcond)

  return reconstruction_matrix
# %%
#setup the basic elements in hcipy. This is where most of the heavy lifting is done; in the setup.
grid_size=120 #define number of pixels across our telescope aperture.
D=1  #define the telescope size in meters.
pupil_grid = make_pupil_grid(grid_size, diameter=D)  #define our aperture grid (pupil grid)
pwfs_grid = make_pupil_grid(120, 195/14)
telescope_aperture = make_circular_aperture(D)  #this is a function that returns a telescope generator. Note this is a function.
telescope_pupil=telescope_aperture(pupil_grid)   #telescope aperture (primary mirror)
plt.figure()
imshow_field(telescope_pupil)  # hcipy has a fancy version of plt.imshow() that are for Fields

#pick our wavelength to use for the simulation
wavelength=750e-9
k=2*np.pi/wavelength #wavenumber. convert between microns & radians wavefront error.
# %%
wf= Wavefront(telescope_pupil,wavelength=wavelength) #electric field in hcipy
wf.total_power = 1
# %%
focal_grid = make_focal_grid(q=4, num_airy=20,spatial_resolution=wavelength/D) # how we want to sample the grid that our psf will be on...think of this like our camera
propagator = FraunhoferPropagator(pupil_grid, focal_grid)  #this encodes our fourier transform as it propagates things from the telescope to our focus.

#reference image and the max for plotting the psf later as well as strehl ratio calculation
im_ref= propagator.forward(wf)
norm= np.max(im_ref.intensity)
# %%
 #okay we are going to make our atmospheric turublence model here using HCIPy.

# let's pick the properties of our turblence
fried_parameter = 0.2  # meter; vary this from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meter (this parameter does not really matter)
vx,vy=10.,0. # windspeed in m/s   vary this
velocity = np.sqrt(vx**2.+vy**2.)  # meter/sec
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2 which our model wants

# make our atmospheric turbulence layer
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
# %%
#make the DM
num_actuators = 12 # set the number of actuators

actuator_spacing = D / num_actuators
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)

# %%
# setup Pyramid WFS
pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, wavelength_0=wavelength)
wfs_camera = NoiselessDetector(pupil_grid)

#commands to modulate the PyWFS, get an image out, and calculate a reference slope
wfs_camera.integrate(pwfs(wf), 1)
img_ref = wfs_camera.read_out()
# %%
#Make command matrix for controller.
#This code will have to be rerun everytime you change a parameter about the PyWFS or DM.
#Just run this line of code & do not peak at the function that does the work for you.
CM=make_command_matrix(deformable_mirror, pwfs, wfs_camera, wf)
# %%
#leaky integrator parameters
gain = 0.3
leakage = 0.999

#AO loop speed: 800Hz
dt=1./800
# %%
num_iterations = 100 #number of iterations in our simulation
sr=[] # so we can find the average strehl ratio

# create figure
fig=plt.figure(figsize=(15,8))

# generate animation object; two optional backends FFMpeg or GifWriter.
anim = FFMpegWriter('AO_simulations_standard.mp4', framerate=3)
#anim = GifWriter('AO_simulations_standard.gif', framerate=3)

layer.t = 0
for timestep in trange(num_iterations):
    #get a clean wavefront
    wf_in=wf.copy()

    #evolve the atmospheric turbulence
    layer.t = timestep*dt

    #pass the wavefront through the turbulence
    wf_after_atmos = layer.forward((wf_in))

    #pass the wavefront through the DM for correction
    wf_after_dm = deformable_mirror.forward(wf_after_atmos)

    #send the wavefront containing the residual wavefront error to the PyWFS and get slopes
    wf_pyr = pwfs.forward(wf_after_dm)

    wfs_camera.integrate(wf_pyr, dt)
    wfs_image = wfs_camera.read_out().astype('float')
    wfs_image /= np.sum(wfs_image)

    diff_image = wfs_image - img_ref
    deformable_mirror.actuators = (1-leakage) * deformable_mirror.actuators - gain * CM.dot(diff_image)

    # Propagate to focal plane
    wf_focal = propagator.forward(wf_after_dm )

    #calculate the strehl ratio to use as a metric for how well the AO system is performing.
    strehl_foc=get_strehl_from_focal(wf_focal.intensity/norm,im_ref.intensity/norm)
    sr.append(strehl_foc)
    #plot the results
    if timestep % 15 == 0: #change this if you want to have more or less frames saved to the image.
        plt.close(fig)
        fig=plt.figure(figsize=(15,8))
        plt.suptitle('Time %.2f s / %d s' % (timestep*dt, dt*num_iterations))

        plt.subplot(1,3,1)
        plt.title('Pyramid image')
        plt.imshow(wfs_image.shaped)
        plt.colorbar()

        plt.subplot(1,3,2)
        plt.title('Residual wavefront error [rad]')
        res=wf_after_dm.phase*telescope_pupil
        imshow_field(res, cmap='RdBu')
        plt.colorbar()

        plt.subplot(1,3,3)
        plt.title('Inst. PSF; Strehl %.2f'% (np.mean(np.asarray(sr))))
        imshow_field(np.log10(wf_focal.intensity/norm), cmap='inferno')
        plt.colorbar()
        plt.subplots_adjust(hspace=0.3)
        anim.add_frame()
#plt.suptitle('Gain = %.2f' % (gain)) # can change this to be the parameter you are varying
#plt.savefig('AO_vary_gain%.2f.png' % (gain)) #example to save the last figure to see how the parameter varied your performance
plt.close()
anim.close()
anim

# plt.plot(np.array(sr))
# plt.show()
# %%
