# %%
import numpy as np
from matplotlib import pyplot as plt
from photonics import LanternReader

img = np.load("../../../muirseal_analysis/data/pl_250304/flat_10ms.npy")
lr = LanternReader(img, "muirseal_250308")
lr.reconstruct_image(img, lr.get_intensities(img))
# %%
plt.imshow(img)
# %%
plt.imshow(lr.reconstruct_image(img, lr.get_intensities(img)))
# %%
