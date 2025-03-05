# %%
from photonics.experiments.lantern_cameras import LanternCamera
import numpy as np
from matplotlib import pyplot as plt
# %%
lc = LanternCamera()
# %%
img = np.load("../muirseal_analysis/data/pl_250304/flat_10ms.npy")
# %%
lc.set_centroids(img)
# %%
x_coords = [255, 297, 254, 215, 210, 249, 285, 341, 324, 293, 245, 207, 175, 168, 221, 264, 303, 332]
y_coords = [309, 292, 265, 283, 325, 352, 338, 299, 259, 236, 225, 241, 270, 313, 386, 389, 372, 340]

# %%
