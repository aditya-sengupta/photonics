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
dm = optics.deformable_mirror
dm.flatten()
for (setname, nv) in zip(["train", "test"], [ntrain, ntest]):
    zamps = np.random.uniform(-1, 1, (nv, nzern))
    current_normalizations = np.sqrt(np.sum(zamps ** 2, axis=1))
    test_normalizations = np.random.uniform(0, 1, (nv,))
    zamps /= current_normalizations[:,np.newaxis]
    zamps *= test_normalizations[:,np.newaxis]
    lantern_coeffs = np.empty((nv, len(lo.lant.init_core_locs)), dtype=np.complex128)

    for (i, c) in enumerate(tqdm(zamps)):
        dm.actuators[:nzern] = c * (optics.wl / (4 * np.pi))
        pupil_wf = dm.forward(optics.wf)
        # assert np.allclose(
            #optics.zernike_basis.coefficients_for(pupil_wf.phase)[:nzern],
            #c
        #)
        focal_wf = optics.focal_propagator.forward(pupil_wf)
        lantern_coeffs[i] = lo.lantern_coeffs(focal_wf)

    np.save(path.join(DATA_PATH, f"sim_trainsets/sim_{setname}set_amplitudes_spieeval.npy"), zamps)
    np.save(path.join(DATA_PATH, f"sim_trainsets/sim_{setname}set_lanterns_spieeval.npy"), lantern_coeffs)

# %%
