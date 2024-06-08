# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
from functools import partial
import numpy as np
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
from photonics.utils import lmap, rms, zernike_names
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.second_stage import correction

# %%
n_filter = 9
f_cutoff = 30
f_loop = 800
dt = 1/f_loop
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
lantern = LanternOptics(optics)
corr = partial(correction, optics=optics, pyramid=pyramid, lantern=lantern, f_loop=f_loop)
focus_ncpa = optics.zernike_to_pupil(2, 0.3)
# %%
optics.turbulence_setup(fried_parameter=1/4, seed=371)
# %%
pyramid_correction = corr(use_pyramid=True, num_iterations=800, ncpa=focus_ncpa, perfect_pyramid=False, perfect_lantern=True)
# %%
multiwfs_correction = corr(use_pyramid=True, use_lantern=True, num_iterations=800, ncpa=focus_ncpa, perfect_pyramid=False, perfect_lantern=True)

# %%
plt.plot(pyramid_correction["time"], pyramid_correction["strehl_ratios"])
plt.plot(multiwfs_correction["time"], multiwfs_correction["strehl_ratios"])
# %%
def plot_reconstruction(correction_result):
    pyramid_readings_in_rad = np.array(correction_result["pyramid_readings"]) / (optics.wl / (4 * np.pi))
    true_phase_zernikes = np.array(correction_result["focal_zernikes_truth"])
    lantern_readings_in_rad = np.array(correction_result["lantern_readings"]) / (optics.wl / (4 * np.pi))
    filtered_readings_in_rad = (np.array(correction_result["lpf_readings"])[:,:lantern.nmodes] + np.array(correction_result["hpf_readings"])) / (optics.wl / (4 * np.pi))
    fig, axs = plt.subplots(3,3, sharex=True, sharey=True)
    pyramid_resid = true_phase_zernikes - pyramid_readings_in_rad
    lantern_resid = true_phase_zernikes[:,:lantern.nmodes] - lantern_readings_in_rad
    filtered_resid = true_phase_zernikes[:,:lantern.nmodes] - filtered_readings_in_rad
    for k in range(9):
        ax = axs[k // 3, k % 3]
        ax.set_title(zernike_names[k])
        ax.plot(pyramid_resid[:,k], label="Pyramid")
        ax.plot(lantern_resid[:,k], label="Lantern")
        ax.plot(filtered_resid[:,k], label="Filtered")
        if k == 3:
            ax.set_ylabel("Truth - WFS reading, rad")
        if k == 7:
            ax.set_xlabel("Iteration number")
    plt.legend(bbox_to_anchor=(0.7, 4.0), ncol=3)

# %%
plot_reconstruction(pyramid_correction)
# %%
plot_reconstruction(multiwfs_correction)

# %%
recon = (np.array(pyramid_correction["lpf_readings"]) + np.array(pyramid_correction["hpf_readings"])) / optics.wl / (4 * np.pi)
# %%
truth = np.array(pyramid_correction["focal_zernikes_truth"]) [:,:9]
# %%
plt.plot(truth - recon)
# %%
