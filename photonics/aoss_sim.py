# helper functions, etc. written by Maaike for AO Summer School that I might adapt

import numpy as np
import hcipy

def optical_setup(config):
    """
    Sets up basic optical elements in hcipy.

    Parameters
    ----------
    config : dict (str -> any), containing at least
        "grid_size" : int
        "D" : float
        "wavelength" : float

    Returns
    -------
    pupil_grid : CartesianGrid
    focal_grid : CartesianGrid
    propagator : FraunhoferPropagator
    wf_init : Wavefront
    """
    grid_size, D, wavelength = config["grid_size"], config["D"], config["wavelength"]
    #setup the basic elements in hcipy. This is where most of the heavy lifting is done; in the setup.
    pupil_grid = hcipy.make_pupil_grid(grid_size, diameter=D)  #define our aperture grid (pupil grid)
    telescope_aperture = hcipy.make_circular_aperture(D)  #this is a function that returns a telescope generator. Note this is a function.
    telescope_pupil = telescope_aperture(pupil_grid)   #telescope aperture (primary mirror)

    #pick our wavelength to use for the simulation
    # k = 2 * np.pi / wavelength #wavenumber. convert between microns & radians wavefront error.
    wf_init = hcipy.Wavefront(telescope_pupil,wavelength=wavelength) #electric field in hcipy
    wf_init.total_power = 1
    focal_grid = hcipy.make_focal_grid(q=4, num_airy=20,spatial_resolution=wavelength/D) # how we want to sample the grid that our psf will be on...think of this like our camera
    propagator = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)  #this encodes our fourier transform as it propagates things from the telescope to our focus.

    #reference image and the max for plotting the psf later as well as strehl ratio calculation
    # im_ref = propagator.forward(wf)
    # norm = np.max(im_ref.intensity)
    return telescope_pupil, pupil_grid, focal_grid, propagator, wf_init

def dm_setup(pupil_grid, config):
    """
    Sets up basic optical elements in hcipy.

    Parameters
    ----------
    config : dict (str -> any), containing at least
        "D" : float
        "num_actuators" : int

    Returns
    -------
    dm : DeformableMirror
    """
    actuator_spacing = config["D"] / config["num_actuators"]
    influence_functions = hcipy.make_gaussian_influence_functions(pupil_grid, config["num_actuators"], actuator_spacing)
    dm = hcipy.DeformableMirror(influence_functions)
    return dm

def make_command_matrix(deformable_mirror, mpwfs, modsteps, wfs_camera, wf,pixels_pyramid_pupils, rcond=1e-3):
  probe_amp = 0.02 * wf.wavelength
  response_matrix = []
  num_modes = deformable_mirror.num_actuators

  for i in range(int(num_modes)):
      slope = 0

      for s in [1, -1]:
          amp = np.zeros((num_modes,))
          amp[i] = s * probe_amp
          deformable_mirror.flatten()
          deformable_mirror.actuators = amp

          dm_wf = deformable_mirror.forward(wf)
          wfs_wf = mpwfs.forward(dm_wf)

          for m in range(modsteps):
                wfs_camera.integrate(wfs_wf[m], 1)

          image_nophot = bin_image(wfs_camera.read_out().shaped, pyr_bin).flatten()
          image_nophot /= image_nophot.sum()
          sxy = pyramid_slopes(image_nophot,pixels_pyramid_pupils)

          slope += s * (sxy-pyr_ref)/(2*probe_amp)  #these are not really slopes; this is just a normalized differential image

      response_matrix.append(slope.ravel())

  response_mtx = hcipy.ModeBasis(response_matrix)

  reconstruction_matrix = hcipy.inverse_tikhonov(response_mtx.transformation_matrix, rcond=rcond)

  return reconstruction_matrix

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


def pyramid_slopes(image, pixels_pyramid_pupils, config):

    ''' Parameters
    ----------
    image : 1D numpy array
         The flatted image of the pyramid wfs pupils

    Returns
    -------
    slopes : 1D numpy array
        x- and y- slopes inside the pupil stacked onto of eachother for 1D array
    '''
    D = config["D"]
    pyramid_plot_grid = hcipy.make_pupil_grid(pixels_pyramid_pupils*2, D) #hardcoded for now/ease

    pyr1 = hcipy.circular_aperture(0.5*D,[-0.25*D,0.25*D])(pyramid_plot_grid)
    pyr2 = hcipy.circular_aperture(0.5*D,[0.25*D,0.25*D])(pyramid_plot_grid)
    pyr3 = hcipy.circular_aperture(0.5*D,[-0.25*D,-0.25*D])(pyramid_plot_grid)
    pyr4 = hcipy.circular_aperture(0.5*D,[0.25*D,-0.25*D])(pyramid_plot_grid)
    N=4*np.sum(pyr1[pyr1>0])
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

