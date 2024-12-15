# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
piston_all = np.load("../../data/pl_241204/piston_all_images.npy")
# %%
for i in range(sweep_all.shape[2]):
    plt.imshow(sweep_all[:,:,i])
    plt.show()
# %%
sweep_all = np.load("../../data/pl_241204/zernike_sweep_images.npy")

# %%
