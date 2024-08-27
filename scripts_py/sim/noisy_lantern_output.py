# %%
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")
import numpy as np
from tqdm import tqdm
from os import path
from hcipy import NoisyDetector, imshow_field, Field, Wavefront
from photonics.utils import DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.utils import date_now, time_now
from matplotlib import pyplot as plt
from scipy.stats import poisson
# %%
exptime_scaling = 2000
optics = Optics(lantern_fnumber=12, dm_basis="modal")
lo = LanternOptics(optics)
# %%
dm = optics.deformable_mirror
pupil_wf = dm.forward(optics.wf)
focal_wf = optics.focal_propagator.forward(pupil_wf)
coeffs_true, _ = lo.lantern_output(focal_wf)
coeffs_true *= np.sqrt(lo.focal_grid.weights) * np.sqrt(exptime_scaling)  # arbitrary scaling to reflect photon count
# has to be constant across f/#
intensities_detector = np.abs(coeffs_true ** 2)
intensities_detector += poisson(intensities_detector).rvs()
plt.scatter(np.arange(19), np.abs(coeffs_true ** 2))
plt.scatter(np.arange(19), intensities_detector)
# %%
