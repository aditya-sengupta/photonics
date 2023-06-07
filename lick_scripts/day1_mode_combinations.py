# %%
import numpy as np
import imageio.v3 as iio
import sys

from scipy.stats import linregress

sys.path.append("..")
from src.lantern_reader import LanternReader

# %%
reader = LanternReader(
    nports = 18,
    cutout_size = 18,
    fwhm = 12,
    threshold = 25,
    ext = "tiff",
    imgshape = (540, 720),
    subdir = "pl_230602/mode_combinations_small"
)

seal_reader = LanternReader(
    nports = 18,
    cutout_size = 20,
    fwhm = 12,
    threshold = 25,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_230525"
)

seal_img = np.flip(seal_reader.read_image("pl_230525_z3_a0.0"), axis=1)
seal_reader.set_centroids(seal_img, min_energy=30000)
img = reader.read_image("mode_combinations_small-06022023182116-0")
reader.set_centroids(img)

# %%
# look at reader.plot_ports() and seal_reader.plot_ports(), find the missing index
missing_idxs = np.array([16]) - 1 # the index on SEAL that isn't present on Shane
# %%
xmap = np.polyfit(np.delete(seal_reader.xc, missing_idxs), reader.xc, deg=7)
ymap = np.polyfit(np.delete(seal_reader.yc, missing_idxs), reader.yc, deg=7)
# %%
reader.xc = np.polyval(xmap, seal_reader.xc)
reader.yc = np.polyval(ymap, seal_reader.yc)
reader.radial_shell = seal_reader.radial_shell
reader.fwhm = 20
reader.plot_ports()
# %%
plt.imshow(img)
# %%
plt.imshow(img * reader.port_mask())
# %%
plt.imshow(reader.reconstruct_image(img, reader.get_intensities(img)))
# %%
