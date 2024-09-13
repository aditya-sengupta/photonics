"""
This script generates training sets for the neural network reconstructor, which is trained on threadripper.
"""
# %%
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")
import numpy as np
from tqdm import tqdm, trange
from os import path
from hcipy import Wavefront, Field, NoisyDetector
from photonics.utils import DATA_PATH
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.utils import date_now, time_now
from matplotlib import pyplot as plt

nzern = 19
lfs = None
# %%
throughputs = []
fs = np.arange(3, 12, step=0.1)
for fnumber in tqdm(fs):
    optics = Optics(lantern_fnumber=fnumber, dm_basis="modal")
    lo = LanternOptics(optics)
    if fnumber ==  3:
        lfs = [np.abs(l).flatten() for l in lo.launch_fields]
    focal_wf = optics.focal_propagator.forward(optics.wf)
    coeffs_true, pl_image = lo.lantern_output(focal_wf)
    # pl_wf = Wavefront(Field(pl_image.flatten(), lo.focal_grid), wavelength=optics.wl)
    # detector = NoisyDetector(detector_grid=lo.focal_grid)
    # detector.integrate(pl_wf, 1e4)
    # measurement = detector.read_out()
    throughputs.append(np.sum(np.abs(pl_image ** 2)) / np.sum(focal_wf.intensity))
    # throughputs.append(np.sum([np.sum(measurement * l) for l in lfs]))
# %%
plt.plot(fs, np.array(throughputs) * 100)
plt.xlabel("f-number")
plt.ylabel("Photonic lantern throughput (%)")
# %%
np.save(path.join(DATA_PATH, "throughput_fs.npy"), fs)
np.save(path.join(DATA_PATH, "throughputs.npy"), throughputs)
# %%
