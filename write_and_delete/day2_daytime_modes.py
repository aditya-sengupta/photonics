# %%
import numpy as np
import imageio.v3 as iio
import os, sys, re

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

sys.path.append("..")
from photonics.experiments.lantern_reader import LanternReader, PROJECT_ROOT

# %%
reader = LanternReader(
    nports = 18,
    cutout_size = 18,
    fwhm = 12,
    threshold = 25,
    ext = "tiff",
    imgshape = (540, 720),
    subdir = "pl_230603/large_mode_combination"
)

img = reader.read_image("day2_large_modecomb-06032023183049-0")
reader.set_centroids(img)
# %%
for subdir in ["large_mode_combination"]:
    pattern = re.compile(r"day2_large_modecomb-06032023(\d{6})-\d+.tiff")
    reader.subdir = "pl_230603/" + subdir
    dirname = os.path.join(PROJECT_ROOT, "data", reader.subdir)
    fnames = [x for x in os.listdir(dirname) if x.endswith("." + reader.ext)]
    intensities = np.zeros((len(fnames), reader.nports))
    timestamps = []
    all_images = np.zeros((len(fnames), img.shape[0], img.shape[1]))
    for (i, fname) in enumerate(tqdm(fnames)):
        timestamps.append(pattern.match(fname)[1])
        all_images[i,:,:] = reader.read_image(os.path.join(dirname, fname[:-5]))

    all_images = all_images[np.argsort(timestamps)]

    diff_intensities = np.array([
        reader.get_intensities(d) for d in np.diff(all_images, axis=0)
    ])
    np.save(os.path.join(dirname, f"{subdir}_diff_intensities.npy"), intensities)
# %%
plt.plot(np.sum(np.abs(diff_intensities)[:50], axis=1))
# %%
