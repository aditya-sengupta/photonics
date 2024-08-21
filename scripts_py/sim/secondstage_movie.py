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
# %%
focus_ncpa = optics.zernike_to_pupil(2, 0.5)

# %%
niter = 200
second_stage_iter = 100
optics.turbulence_setup(fried_parameter=optics.telescope_diameter/4, seed=371)
twostage_correction = corr(use_pyramid=True, use_lantern=True, num_iterations=niter, ncpa=focus_ncpa, pyramid_recon="linear", lantern_recon="nn", second_stage_iter=second_stage_iter)

# %%
psfs = np.array([x.intensity.shaped for x in twostage_correction["point_spread_functions"]])
np.save(DATA_PATH + "/multiwfs_psfs.npy", psfs)
# %%
sr = twostage_correction["strehl_ratios"]
np.save(DATA_PATH + "/multiwfs_anim_strehls.npy", sr)
# %%
