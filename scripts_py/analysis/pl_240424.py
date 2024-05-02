"""
This script demonstrates that we were able to recover linear ranges using the photonic lantern on the ShaneAO calibration source on 2024-04-24.
"""

# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from photonics.utils import PROJECT_ROOT
from photonics.decometify import intensities_from_comet
from photonics.linearity import make_interaction_matrix, make_linearity, plot_linearity, interpolate_weights, linearity_loss
from tqdm import tqdm

ampere2rad = 2 * np.pi * 0.51255 / 1.55  

# %%
directory_240424 = PROJECT_ROOT + "/data/pl_240424"
dmc_files = np.sort(list(filter(lambda x: x.startswith("dmc_240425"), os.listdir(directory_240424))))
img_files = list(map(lambda x: x.replace("dmc_240425", "pl_240425"), dmc_files))

# %%
# On Night 1, I ran 40 successful experiments
# 1 test sweep over Zernikes
# 36 full sweeps over Zernikes (12 modes, 3 times each)
# 2 random combinations
# 1 mixed modes
all_dmcs = []
all_intensities = []
for (d_f, i_f) in zip(tqdm(dmc_files), img_files):
    imgs = np.load(directory_240424 + "/" + i_f)
    if imgs.shape[1] == 520:
        dmc = np.load(directory_240424 + "/" + d_f)
        intensities = list(map(intensities_from_comet, imgs))
        all_dmcs.append(dmc)
        all_intensities.append(intensities)
        
# %%
# First, let's try and identify when we have full linearity sweeps in contiguous sets

# The first one is 1:13
# The second one is 13:25 - throwing this one out because it seems to be worse for some reason
# The third one is 28:40

# This array is (zernike number, amplitude index, port number)
amplitudes = np.arange(-1, 1, 0.1) * ampere2rad
mode_sweeps = ((np.array(all_intensities[1:13]) + np.array(all_intensities[28:40])) / 2)[:,:-1,:]

# ((np.array(all_intensities[1:13]) + np.array(all_intensities[13:25]) + np.array(all_intensities[28:40])) / 3)[:,:-1,:]

# %%
idx_zero, w_zero = interpolate_weights(amplitudes, 0.0)
flat = w_zero * mode_sweeps[0,idx_zero,:] + (1 - w_zero) * mode_sweeps[0,idx_zero+1,:]

# %%
shane_modes = ["focus", "astig", "astig45", "tricoma", "tricoma60", "coma", "coma90", "spherical44-", "spherical44+", "spherical40", "spherical42+", "spherical42-"]

for nmodes in [3, 7, 12]:
    im = make_interaction_matrix(amplitudes, mode_sweeps[:nmodes,:,:], poke_amplitude=0.1)
    cm = np.linalg.pinv(im, rcond=1e-3)
    responses = make_linearity(amplitudes, mode_sweeps, cm)
    plot_linearity(amplitudes, responses, title_mod="ShaneAO calibration source, 2024-04-25", zlabels=shane_modes)

# %%
linearity_loss(amplitudes, responses)
# %%
# now let's look at the random combinations
recons = []
experiment_idx = 26
for (dmc, intensity) in zip(all_dmcs[experiment_idx], all_intensities[experiment_idx]):
    recons.append(cm @ (intensity - flat))
    
recons = np.array(recons)
zidx = 0
perm = np.argsort(all_dmcs[experiment_idx][:,zidx])
plt.scatter(np.arange(100), all_dmcs[experiment_idx][:,zidx][perm], label=f"Applied {shane_modes[zidx]}", s=4)
plt.scatter(np.arange(100), recons[:,zidx][perm], label="Reconstructed", s=4)
plt.legend()
plt.show()
# %%

