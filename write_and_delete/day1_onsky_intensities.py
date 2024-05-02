# %%
import numpy as np
import imageio.v3 as iio
import os, sys

import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

sys.path.append("..")
from photonics.lantern_reader import LanternReader, PROJECT_ROOT

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

img = reader.read_image("mode_combinations_small-06022023182116-0")
reader.set_centroids(img)

# look at reader.plot_ports() and seal_reader.plot_ports(), find the missing index
missing_idx = 16 - 1 # the index on SEAL that isn't present on Shane
reader.add_port(445, 265)

reader.subdir = "pl_230602/onsky4_part1"
"""
reader.xc *= 2
reader.yc *= 2
reader.fwhm *= 2
reader.xc += 10
reader.plot_ports()
"""
reader.xc += 3
reader.xc[4] += 10
reader.xc[5] += 5
reader.xc[15] += 5
# %%
img_onsky = reader.read_image("onsky34-06022023224826-0")
plt.imshow(img_onsky * reader.port_mask())
# %%
for subdir in ["onsky1"]:
    reader.subdir = "pl_230602/" + subdir
    dirname = os.path.join(PROJECT_ROOT, "data", reader.subdir)
    fnames = list(filter(lambda x: x.endswith(".tiff"), os.listdir(dirname)))
    intensities = np.zeros((len(fnames), reader.nports))
    for (i, fname) in enumerate(tqdm(fnames)):
        img = reader.read_image(os.path.join(dirname, fname[:-5]))
        intensities[i,:] = reader.get_intensities(img)

    np.save(os.path.join(dirname, f"{subdir}_intensities.npy"), intensities)
# %%
np.load(os.path.join(dirname, "onsky4_part1_intensities.npy"), intensities)

# %%
