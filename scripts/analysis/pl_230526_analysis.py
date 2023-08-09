# %%
import os, sys, re, tqdm
sys.path.append("..")
from photonics.lantern_reader import LanternReader

reader = LanternReader(
    nports = 18,
    cutout_size = 20,
    fwhm = 18,
    ext = "pgm",
    imgshape = (1800, 1920)
)

img = reader.read_image("pl_230526/pl_230526_flat")
reader.set_centroids(img, min_energy=25000)
plt.imshow(img)
plt.show()
plt.imshow(reader.reconstruct_image(img, reader.get_intensities(img)))

# %%
img0 = reader.read_image("pl_230526/pl_230526_flat")
# %%
plt.imshow(img - img0)
plt.colorbar()
# %%
