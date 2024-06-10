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
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from photonics.utils import lmap, rms, zernike_names, DATA_PATH
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
niter = 100
second_stage_iter = 20
D_over_r0s = [16]
lantern_recons = ["linear", "gs"]
strehls_grid = np.zeros((len(D_over_r0s), len(lantern_recons), niter))
for (i, D_over_r0) in enumerate(D_over_r0s):
    optics.turbulence_setup(fried_parameter=optics.telescope_diameter/D_over_r0, seed=371)
    for (j, lantern_recon) in enumerate(lantern_recons):
        print(f"D/r0 = {D_over_r0}, {lantern_recon}")
        use_lantern = (lantern_recon != "none")
        twostage_correction = corr(use_pyramid=True, use_lantern=use_lantern, num_iterations=niter, ncpa=focus_ncpa, pyramid_recon="linear", lantern_recon=lantern_recon, second_stage_iter=second_stage_iter)
        strehls_grid[i,j,:] = twostage_correction["strehl_ratios"]

# %%
plt.plot(strehls_grid[0,0,:], label="linear")
plt.plot(strehls_grid[0,1,:], label="G-S")
plt.xlabel("Frame number")
plt.ylabel("Strehl ratio")
# %%
gs_zernikes_over_time = np.array([optics.zernike_basis.coefficients_for(x.phase) for x in twostage_correction["wavefronts_after_dm"]])
# %%
lantern_measured_zernikes = np.array(twostage_correction["lantern_readings"])
# %%
plt.plot(np.array(twostage_correction["focal_zernikes_truth"])[:,2], label="True focal-plane focus")
plt.plot(gs_zernikes_over_time[:,2], label="True pupil-plane focus")
plt.plot(lantern_measured_zernikes[:,2] / (optics.wl / (4 * np.pi)), label="Lantern-reconstructed focal-plane focus")
plt.legend()
# %%
plt.figure()
im = imshow_field(twostage_correction["wavefronts_after_dm"][0].phase)
for i in range(0, 100):
    imshow_field(twostage_correction["wavefronts_after_dm"][i].phase, vmin=(-np.pi), vmax=(np.pi))
    plt.pause(0.5)
    plt.draw()
# %%
