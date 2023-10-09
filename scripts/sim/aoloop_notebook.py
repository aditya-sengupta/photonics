# %%
from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# %%
#@title
 #helper functions

def bin_image(imin,fbin):

    ''' Parameters
    ----------
    imin : 2D numpy array
         The 2D image that you want to bin
    fbin : int


    Returns
    -------
    out : 2D numpy array
        the 2D binned image
        '''
    out=np.zeros((int(imin.shape[0]/fbin),int(imin.shape[1]/fbin)))
   #  begin binning
    for i in np.arange(fbin-1,imin.shape[0]-fbin,fbin):
        for j in np.arange(fbin-1,imin.shape[1]-fbin,fbin):
            out[int((i+1)/fbin)-1,int((j+1)/fbin)-1]=np.sum(imin[i-int((fbin-1)/2):i+int((fbin-1)/2),j-int((fbin-1)/2):j+int((fbin-1)/2)])
    return out


def pyramid_slopes(image,pixels_pyramid_pupils):

    ''' Parameters
    ----------
    image : 1D numpy array
         The flatted image of the pyramid wfs pupils

    Returns
    -------
    slopes : 1D numpy array
        x- and y- slopes inside the pupil stacked onto of eachother for 1D array
        '''
    D=1 #hardcoded for now/ease
    pyramid_plot_grid = make_pupil_grid(pixels_pyramid_pupils*2, D) #hardcoded for now/ease

    pyr1=circular_aperture(0.5*D,[-0.25*D,0.25*D])(pyramid_plot_grid)
    pyr2=circular_aperture(0.5*D,[0.25*D,0.25*D])(pyramid_plot_grid)
    pyr3=circular_aperture(0.5*D,[-0.25*D,-0.25*D])(pyramid_plot_grid)
    pyr4=circular_aperture(0.5*D,[0.25*D,-0.25*D])(pyramid_plot_grid)
    N=4*np.sum(pyr1[pyr1>0])
    norm=(image[pyr1>0]+image[pyr2>0]+image[pyr3>0]+image[pyr4>0])/N
    sx=(image[pyr1>0]-image[pyr2>0]+image[pyr3>0]-image[pyr4>0])
    sy=(image[pyr1>0]+image[pyr2>0]-image[pyr3>0]-image[pyr4>0])
    return np.array([sx,sy]).flatten()

def plot_slopes(slopes,pixels_pyramid_pupils):
    '''
    Only want if we decide to plot the slopes.

    Parameters
    ----------
    slopes : 1D numpy array
         The flatted slopes produced by pyramid_slopes().

    Returns
    -------
    slopes : 1D numpy array
        x- and y- slopes mapped within their pupils for easy plotting
    '''
    D=1
    mid=int(slopes.shape[0]/2)
    pyramid_plot_grid = make_pupil_grid(pixels_pyramid_pupils, D) #hardcoded for now/ease
    pyr_mask=circular_aperture(D)(pyramid_plot_grid)
    sx=pyr_mask.copy()
    sy=pyr_mask.copy()
    sx[sx>0]=slopes[0:mid]
    sy[sy>0]=slopes[mid::]
    return [sx,sy]


# %%
#@title

def make_command_matrix(deformable_mirror, mpwfs,modsteps,wfs_camera,wf,pixels_pyramid_pupils):

  probe_amp = 0.02 * wf.wavelength
  response_matrix = []
  num_modes=deformable_mirror.num_actuators

  for i in range(int(num_modes)):
      slope = 0

      for s in [1, -1]:
          amp = np.zeros((num_modes,))
          amp[i] = s * probe_amp
          deformable_mirror.flatten()
          deformable_mirror.actuators = amp

          dm_wf = deformable_mirror.forward(wf)
          wfs_wf = mpwfs.forward(dm_wf)

          for m in range (modsteps) :
                wfs_camera.integrate(wfs_wf[m], 1)

          image_nophot = bin(wfs_camera.read_out().shaped,pyr_bin).flatten()
          image_nophot/=image_nophot.sum()
          sxy=pyramid_slopes(image_nophot,pixels_pyramid_pupils)

          slope += s * (sxy-pyr_ref)/(2*probe_amp)  #these are not really slopes; this is just a normalized differential image

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
telescope_aperture  = circular_aperture(D)  #this is a function that returns a telescope generator. Note this is a function.
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
fried_parameter = 0.12  # meter; vary this from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meter (this parameter does not really matter)
vx,vy=10.,0. # windspeed in m/s   vary this
velocity = np.sqrt(vx**2.+vy**2.)  # meter/sec
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2 which our model wants

# make our atmospheric turbulence layer
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
# %%
#make the DM
num_actuators = 9 # set the number of actuators

actuator_spacing = D / num_actuators
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)

# %%
# setup Pyramid WFS
pixels_pyramid_pupils=10 # number of pixels across the pupil; want 120 %(mod) pixels_pyramid_pupils =0
pupil_grid_diameter = grid_size/56 * D
pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)
mld=5 # modulation radius in lambda/D
modradius = mld*wavelength/D # modulation radius in radians;
modsteps = 12 #keep this as a factor of 4. Significantly increases computation time.
pwfs_grid = make_pupil_grid(grid_size, 2 * pupil_grid_diameter)
pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, wavelength_0=wavelength)
wfs_camera = NoiselessDetector(pupil_grid)

#bin our pyramid image
pyramid_plot_grid = make_pupil_grid(pixels_pyramid_pupils*2, D)
pyr_bin=int((grid_size*2)/(2*pixels_pyramid_pupils))

#commands to modulate the PyWFS, get an image out, and calculate a reference slope
for m in range(modsteps):
      wfs_camera.integrate(pwfs(wf), 1)
pyr_ref = bin_image(wfs_camera.read_out().shaped,pyr_bin).flatten()
pyr_ref=pyramid_slopes(pyr_ref/pyr_ref.sum(),pixels_pyramid_pupils)
# %%
