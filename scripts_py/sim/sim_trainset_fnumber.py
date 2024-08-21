"""
This script generates training sets for the neural network reconstructor, which is trained on threadripper.
"""
# %%
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")
import numpy as np
from tqdm import tqdm
from os import path
from photonics.utils import DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics

optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
lo = LanternOptics(optics)
nzern = 9
N = 60_000
ntrain = 4 * N // 5
ntest = N - ntrain
# %%
zamps = {}
for (setname, nv) in zip(["train", "test"], [ntrain, ntest]):
    zamps_v = np.random.uniform(-1, 1, (nv, nzern))
    current_normalizations = np.sqrt(np.sum(zamps_v ** 2, axis=1))
    test_normalizations = np.random.uniform(0, 1, (nv,))
    zamps_v /= current_normalizations[:,np.newaxis]
    zamps_v *= test_normalizations[:,np.newaxis]
    zamps[setname] = zamps_v
# %%
dm = optics.deformable_mirror
dm.flatten()
for fnumber in range(3, 12):
    optics = Optics(lantern_fnumber=fnumber, dm_basis="modal")
    lo = LanternOptics(optics)
    for (setname, nv) in zip(["train", "test"], [ntrain, ntest]):
        lantern_coeffs = np.empty((nv, len(lo.lant.init_core_locs)), dtype=np.complex128)
        zamps_v = zamps[setname]
        for (i, c) in enumerate(tqdm(zamps_v)):
            dm.actuators[:nzern] = c * (optics.wl / (4 * np.pi))
            pupil_wf = dm.forward(optics.wf)
            focal_wf = optics.focal_propagator.forward(pupil_wf)
            lantern_coeffs[i] = lo.lantern_coeffs(focal_wf)

        np.save(path.join(DATA_PATH, f"sim_trainsets/sim_{setname}set_amplitudes_fnumber_{fnumber}.npy"), zamps)
        np.save(path.join(DATA_PATH, f"sim_trainsets/sim_{setname}set_lanterns_fnumber_{fnumber}_dt_{date_now()}{time_now()}.npy"), lantern_coeffs)

# %%
