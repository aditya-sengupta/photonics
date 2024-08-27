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
from hcipy import Wavefront, Field, NoisyDetector
from photonics.utils import DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.utils import date_now, time_now

optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
lo = LanternOptics(optics)
nzern = 19
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
lfs = [np.abs(l).flatten() for l in lo.launch_fields]
# %%
for fnumber in range(3, 12):
    print(fnumber)
    optics = Optics(lantern_fnumber=fnumber, dm_basis="modal")
    lo = LanternOptics(optics)
    for (setname, nv) in zip(["train", "test"], [ntrain, ntest]):
        lantern_coeffs = np.empty((nv, len(lo.lant.init_core_locs)), dtype=np.complex128)
        zamps_v = zamps[setname]
        for (i, c) in enumerate(tqdm(zamps_v)):
            dm.actuators[:nzern] = c * (optics.wl / (4 * np.pi))
            pupil_wf = dm.forward(optics.wf)
            focal_wf = optics.focal_propagator.forward(pupil_wf)
            coeffs_true, pl_image = lo.lantern_output(focal_wf)
            pl_wf = Wavefront(Field(pl_image.flatten(), lo.focal_grid), wavelength=optics.wl)
            detector = NoisyDetector(detector_grid=lo.focal_grid)
            detector.integrate(pl_wf, 1e4)
            measurement = detector.read_out()
            lantern_coeffs[i] = np.array([np.sum(measurement * l) for l in lfs])

        np.save(path.join(DATA_PATH, f"sim_trainsets/fnumber/sim_{setname}set_amplitudes_fnumber_{fnumber}_date_{date_now()}.npy"), zamps_v)
        np.save(path.join(DATA_PATH, f"sim_trainsets/fnumber/sim_{setname}set_lanterns_fnumber_{fnumber}_date_{date_now()}.npy"), lantern_coeffs)

# %%
