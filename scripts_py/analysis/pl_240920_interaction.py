# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import trange
from photonics.experiments.shane import ShaneLantern
from photonics.linearity import make_linearity, plot_linearity, make_interaction_matrix

# %%
flats = []
with h5py.File("../../data/pl_240920/sweep_all_modes_2024-09-20T17.39.53.145005.hdf5") as f:
    for (row_dmc, row_intensity) in zip(np.array(f["dmc"]), np.array(f["pl_intensities"])):
        if np.allclose(row_dmc, 0.0):
            flats.append(row_intensity / np.sum(row_intensity))
            
flats = np.array(flats)
flat = np.mean(flats, axis=0)
# %%
intensities = None
with h5py.File("../../data/pl_240920/sweep_all_modes_2024-09-20T17.39.53.145005.hdf5") as f:
    intensities = np.array(f["pl_intensities"])
    
# intensities /= np.sum(intensities, axis=1)[:,np.newaxis]
intensities = intensities - flat

intensities = intensities[:-1,:]
intensities = np.reshape(intensities, (12, 13, 19))

# %%
amplitudes = np.arange(-0.06, 0.0601, step=0.01)
im = make_interaction_matrix(amplitudes, intensities, poke_amplitude=0.01)
cm = np.linalg.pinv(im, rcond=1e-3)

responses = make_linearity(amplitudes, intensities, cm)
plot_linearity(amplitudes, responses)
# %%
im = make_interaction_matrix(amplitudes, intensities[:3,:,:], poke_amplitude=0.01)
cm = np.linalg.pinv(im, rcond=1e-3)

responses = make_linearity(amplitudes, intensities, cm)
plot_linearity(amplitudes * 16, responses * 16, zlabels=["Focus", "Astig", "Astig45"], title_mod="(rad)")
# %%
saved_cmdmat = None
with h5py.File("../../data/pl_240920/intcmd_2024-09-20T19.57.45.761549.hdf5") as f:
    saved_cmdmat = np.array(f["cmdmat"])
    
responses = make_linearity(amplitudes, intensities, saved_cmdmat)
plot_linearity(amplitudes * 16, responses * 16, zlabels=["Focus", "Astig", "Astig45"])
# %%
