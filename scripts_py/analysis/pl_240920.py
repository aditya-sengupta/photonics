# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import trange
from photonics.experiments.shane import ShaneLantern

# %%
with h5py.File("../../data/pl_240920/closedloop_2024-09-20T03.13.11.082069.hdf5") as f:
    cl_attempt_images = np.array(f["lantern_images"])
    spot_radius_px = f["lantern_images"].attrs["spot_radius_px"]
    exp_ms = f["dmcs"].attrs["exp_ms"]
    gain = f["dmcs"].attrs["gain"]
    
# %%
with h5py.File("../../data/pl_240920/centroids_2024-09-20T01.51.33.430571.hdf5") as f:
    centroids = np.array(f["centroids"])
# %%
plt.imshow(cl_attempt_images[0])
# %%
lant = ShaneLantern()
proportions_explained = []
dark = np.load("../../data/pl_240920/dark_exptime_ms_50.0_gain_18.0.npy")
images = [cl_attempt_images[i] + dark for i in range(20)]
for spot_radius_px in trange(50):
    lant.spot_radius_px = spot_radius_px
    lant.set_centroids(centroids, save=False)
    proportions_explained.append([float(np.sum(img * reduce(np.logical_or, lant.masks)) / np.sum(img)) for img in images])
    
# %%
plt.plot(proportions_explained)
# %%
