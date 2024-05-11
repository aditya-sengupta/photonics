"""
This script demonstrates that we were able to recover linear ranges using the photonic lantern on the ShaneAO calibration source on 2024-04-25.
"""

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from os import path
from photonics.utils import PROJECT_ROOT
from photonics.experiments.decometify import intensities_from_comet
from photonics.linearity import make_interaction_matrix, make_linearity, plot_linearity, interpolate_weights, linearity_loss
from tqdm import tqdm

# %%
directory_240424 = PROJECT_ROOT + "/data/pl_240424"
directory_240425 = PROJECT_ROOT + "/data/pl_240425"
dmc_files = np.sort(list(filter(lambda x: x.startswith("dmc_240425"), os.listdir(directory_240425))))
img_files = list(map(lambda x: x.replace("dmc_240425", "pl_240425"), dmc_files))

# %%
# I ran 22 experiments on day 2, but some of these are full Zernike sweeps
# 0, 1: test sweeps over focus
# 2, 3: sweeps over focus (-1, 1, 0.01)
# 4, 5: all zeros
# 6: sweep over all modes, (-0.25, 0.25, 0.01)
# 7: sweep over all modes, (-1, 0.25, 0.05)
# 8: sweep over all modes, (0.25, 1, 0.05)
# 9: mixed modes with 0.01 amplitude
# 10: mixed modes with -0.01 amplitude
# 11: mixed modes with 0.05 amplitude
# 12: mixed modes with -0.05 amplitude
# 13: mixed modes with 0.1 amplitude
# 14: mixed modes with -0.1 amplitude
# 15: sweep over all modes, (-1, 1, 0.05)
# 16: sweep over all modes, (-0.25, 0.25, 0.01)
# 17: 100 random combinations (-0.1, 0.1)
# 18: 100 random combinations (-0.05, 0.05)
# 19: sweep over all modes, (-0.5, 0.5, 0.05)
# 20: sweep over all modes, (-0.5, 0.24, 0.02)
# 21: sweep over all modes, (0.24, 0.52, 0.02)

# I also have on-sky interaction matrices that I can correlate later.
# %%
dmcs = np.load(path.join(directory_240425, dmc_files[7]))
imgs = np.load(path.join(directory_240425, img_files[7]))
throughput = np.sum(np.maximum(imgs, 0), axis=(1,2))
# %%
dmcs_in_rad = dmcs * 40 * (2 * np.pi / 1.55)
dmcs_rms = np.sqrt(np.sum(dmcs_in_rad ** 2, axis=1))

# %%
perm = np.argsort(dmcs_rms)
plt.scatter(dmcs_rms[perm], throughput[perm])
# %%
# first, let's answer: can we get a linear range with just these sweeps?
# This array is (zernike number, amplitude index, port number)
intensities_6 = np.array(list(map(intensities_from_comet, np.load(path.join(directory_240425, img_files[19])))))
# %%
amplitudes = np.arange(-0.5, 0.5001, 0.05)
namp = len(amplitudes)
mode_sweeps = intensities_6[:-1,:]
mode_sweeps = np.array([mode_sweeps[i*namp:(i+1)*namp,:] for i in range(12)])
# ((np.array(all_intensities[1:13]) + np.array(all_intensities[13:25]) + np.array(all_intensities[28:40])) / 3)[:,:-1,:]

# %%
idx_zero, w_zero = interpolate_weights(amplitudes, 0.0)
flat = w_zero * mode_sweeps[0,idx_zero,:] + (1 - w_zero) * mode_sweeps[0,idx_zero+1,:]

# %%
shane_modes = ["focus", "astig", "astig45", "tricoma", "tricoma60", "coma", "coma90", "spherical44-", "spherical44+", "spherical40", "spherical42+", "spherical42-"]

for nmodes in [3, 7, 12]:
    im = make_interaction_matrix(amplitudes, mode_sweeps[:nmodes,:,:], poke_amplitude=0.05)
    cm = np.linalg.pinv(im, rcond=1e-3)
    responses = make_linearity(amplitudes, mode_sweeps, cm)
    plot_linearity(amplitudes, responses, title_mod="ShaneAO calibration source, 2024-04-25", zlabels=shane_modes)

# %%
dmc_files_1 = np.sort(list(filter(lambda x: x.startswith("dmc_240425"), os.listdir(directory_240424))))
img_files_1 = list(map(lambda x: x.replace("dmc_240425", "pl_240425"), dmc_files_1))
# %%
day1_image = np.load(path.join(directory_240424, img_files_1[5]))[14,:]
# %%
day2_image = np.load(path.join(directory_240425, img_files[15]))[28,:]
# %%
