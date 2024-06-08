# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")

import numpy as np
import hcipy as hc
from copy import copy
from matplotlib import pyplot as plt
from hcipy import imshow_field
from photonics.utils import DATA_PATH
from photonics.utils import rms, nanify, lmap
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.second_stage import correction
from juliacall import Main as jl
jl.seval("using Flux")
jl.seval("using JLD2")
# %%
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
lo = LanternOptics(optics)
model_state = jl.JLD2.load(DATA_PATH + f"/pl_nn/pl_nn_spieeval.jld2", "model_state")
nzern = 9
model = jl.Chain(
    jl.Dense(19, 2000, jl.relu),
    jl.Dense(2000, 100, jl.relu),
    jl.Dense(100, 9)
)

jl.Flux.loadmodel_b(model, model_state)
xmin, xmax = (lambda x: (np.min(x), np.max(x)))(np.abs(np.load(DATA_PATH + "/sim_trainsets/sim_trainset_amplitudes_spieeval.npy")) ** 2)
ymin, ymax = (lambda x: (np.min(x), np.max(x)))(np.abs(np.load(DATA_PATH + "/sim_trainsets/sim_trainset_lanterns_spieeval.npy")) ** 2)
zernike_basis = hc.mode_basis.make_zernike_basis(nzern, optics.telescope_diameter, optics.pupil_grid)
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
    norm_intensities = ((intensities - ymin) / (ymax - ymin)).astype(np.float32)
    reconstructed_zernike_coeffs = np.array(model(norm_intensities)) * (xmax - xmin) + xmin
    return reconstructed_zernike_coeffs
        
# %%
nn_inject_recover(5, -0.8)
# %%
test_amplitudes = np.load(DATA_PATH + "/sim_trainsets/sim_testset_amplitudes_spieeval.npy")
test_intensities = np.abs(np.load(DATA_PATH + "/sim_trainsets/sim_testset_lanterns_spieeval.npy")) ** 2
norm_test_intensities = ((test_intensities - ymin) / (ymax - ymin)).astype(np.float32)

squared_error = 0
for (av, lv) in zip(test_amplitudes, norm_test_intensities):
    rv = np.array(model(lv)) * (xmax - xmin) - xmin
    squared_error += np.sum((av - rv) ** 2)

rms_error = np.sqrt(squared_error / test_amplitudes.shape[0])
print(rms_error)
# %%

# %%
