# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")

import numpy as np
import hcipy as hc
from matplotlib import pyplot as plt
from photonics.utils import DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
from juliacall import Main as jl
jl.seval("using Flux")
jl.seval("using JLD2")
# %%
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
lo = LanternOptics(optics)
dm = optics.deformable_mirror

def nn_inject_recover(zernikes, amplitudes):
    dm.flatten()
    if isinstance(zernikes, int):
        zernikes = [zernikes]
    if isinstance(amplitudes, float):
        amplitudes = [amplitudes]
    for (z, a) in zip(zernikes, amplitudes):
        dm.actuators[z] = a * (optics.wl / (4 * np.pi))
    pupil_wf = dm.forward(optics.wf)
    psf = optics.focal_propagator(pupil_wf)
    post_lantern_coeffs = lo.lantern_coeffs(psf)
    intensities = np.abs(post_lantern_coeffs) ** 2
    return lo.nn_reconstruct(intensities)
        
# %%
nn_inject_recover(5, -0.3)
# %%