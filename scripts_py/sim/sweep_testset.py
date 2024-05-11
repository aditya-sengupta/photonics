# %%
from photonics.simulations.lantern_optics import LanternOptics
import numpy as np
from tqdm import trange
from photonics.utils import DATA_PATH
from os import path

lo = LanternOptics(f_number=6.5)
nzern = 19
# %%
amplitudes = np.arange(-1, 1.01, 0.01)
sweep = np.empty((nzern, len(amplitudes), len(lo.lant.init_core_locs)), dtype=np.complex128)
for j in trange(1, nzern + 1):
    for (k, a) in enumerate(amplitudes):
        sweep[j-1,k,:] = lo.lantern_coeffs(lo.zernike_to_focal(j, a))

np.save(path.join(DATA_PATH, "sweep_testset_amplitudes_19.npy"), amplitudes)
np.save(path.join(DATA_PATH, "sweep_testset_lanterns_19.npy"), sweep)
# %%
