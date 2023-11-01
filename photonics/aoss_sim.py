# helper functions, etc. written by Maaike for AO Summer School that I might adapt

import numpy as np
import hcipy
from tqdm import trange

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

def make_command_matrix(deformable_mirror, pwfs, wfs_camera, wf, rcond=1e-3):
  probe_amp = 0.02 * wf.wavelength
  response_matrix = []
  num_modes = deformable_mirror.num_actuators

  for i in trange(num_modes):
      slope = 0

      for s in [1, -1]:
          amp = np.zeros((num_modes,))
          amp[i] = s * probe_amp
          deformable_mirror.flatten()
          deformable_mirror.actuators = amp

          dm_wf = deformable_mirror.forward(wf)
          wfs_wf = pwfs.forward(dm_wf)

          wfs_camera.integrate(wfs_wf[m], 1)

          image_nophot = wfs_camera.read_out()
          image_nophot /= image_nophot.sum()
          slope += s * (image-image_ref)/(2 * probe_amp)

      response_matrix.append(slope)

  response_mtx = hcipy.ModeBasis(response_matrix)

  reconstruction_matrix = hcipy.inverse_tikhonov(response_mtx.transformation_matrix, rcond=rcond)

  return reconstruction_matrix

