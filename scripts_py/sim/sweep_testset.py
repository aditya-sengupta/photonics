# %%
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
import numpy as np
from tqdm import trange
from photonics.utils import DATA_PATH
from os import path

optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
lo = LanternOptics(optics)
dm = optics.deformable_mirror
nzern = 19
# %%
amplitudes = np.arange(-1, 1.01, 0.01)
sweep = np.empty((nzern, len(amplitudes), len(lo.lant.init_core_locs)), dtype=np.complex128)
for j in trange(nzern):
    dm.flatten()
    for (k, a) in enumerate(amplitudes):
        dm.actuators[j] = a * (optics.wl / (4 * np.pi))
        psf = optics.focal_propagator(dm.forward(optics.wf))
        sweep[j,k,:] = lo.lantern_coeffs(psf)

np.save(path.join(DATA_PATH, "sweep_testset_amplitudes_19.npy"), amplitudes)
np.save(path.join(DATA_PATH, "sweep_testset_lanterns_19.npy"), sweep)
# %%
