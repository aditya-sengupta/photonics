"""
This script generates training sets for the neural network reconstructor, which is trained on threadripper.
"""
# %%
from photonics.simulations.lantern_optics import LanternOptics
import numpy as np
from tqdm import tqdm
from photonics.utils import DATA_PATH, datetime_now
from os import path

lo = LanternOptics(f_number=6.5)
nzern = 19
N = 60_000

for lim in [0.12, 0.25, 0.5, 1.0]:
    zcoeffs = np.arange(1, nzern + 1)
    zamps = np.random.uniform(-lim, lim, (N, nzern))
    lantern_coeffs = np.empty((N, len(lo.lant.init_core_locs)), dtype=np.complex128)

    for (i, c) in enumerate(tqdm(zamps)):
        lantern_coeffs[i] = lo.lantern_coeffs(lo.zernike_to_focal(zcoeffs, c))

    dtnow = datetime_now()
    np.save(path.join(DATA_PATH, f"sim_trainsets/sim_trainset_amplitudes_{dtnow}.npy"), zamps)
    np.save(path.join(DATA_PATH, f"sim_trainsets/sim_trainset_lanterns_{dtnow}.npy"), lantern_coeffs)

# %%
