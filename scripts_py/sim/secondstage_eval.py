# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")

# %%
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from photonics.utils import date_now, DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.second_stage import correction

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# %%
n_filter = 9
f_cutoff = 30
f_loop = 800
dt = 1/f_loop
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
lantern = LanternOptics(optics)
corr = partial(correction, optics=optics, pyramid=pyramid, lantern=lantern, f_loop=f_loop, f_cutoff=f_cutoff)
focus_ncpa = optics.zernike_to_pupil(2, 0.3)

# %%
niter = 800
second_stage_iter = 100
D_over_r0s = [64]
lantern_recons = ["none", "linear", "nn", "gs"]
strehls_grid = np.zeros((len(D_over_r0s), len(lantern_recons), niter))
for (i, D_over_r0) in enumerate(D_over_r0s):
    optics.turbulence_setup(fried_parameter=optics.telescope_diameter/D_over_r0, seed=1)
    for (j, lantern_recon) in enumerate(lantern_recons):
        print(f"D/r0 = {D_over_r0}, {lantern_recon}")
        use_lantern = (lantern_recon != "none")
        twostage_correction = corr(use_pyramid=True, use_lantern=use_lantern, num_iterations=niter, ncpa=focus_ncpa, pyramid_recon="linear", lantern_recon=lantern_recon, second_stage_iter=second_stage_iter)
        strehls_grid[i,j,:] = twostage_correction["strehl_ratios"]

np.save(DATA_PATH + f"/strehls_grid_{date_now()}_64.npy", strehls_grid)
# %%
