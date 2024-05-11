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
    pattern = re.compile(r"rnday2_batch2(_cont|)-06032023(\d{6})-\d+.tiff")
    reader.subdir = "pl_230603/" + subdir
    dirname = os.path.join(PROJECT_ROOT, "data", reader.subdir)
    fnames = [x for x in os.listdir(dirname) if x.endswith("." + reader.ext)]
    intensities = np.zeros((len(fnames), reader.nports))
    timestamps = []
    for (i, fname) in enumerate(tqdm(fnames)):
        timestamps.append(pattern.match(fname)[2])
        img = reader.read_image(os.path.join(dirname, fname[:-5]))
        intensities[i,:] = reader.get_intensities(img)

    intensities = intensities[np.argsort(timestamps)]
    np.save(os.path.join(dirname, f"{subdir}_intensities.npy"), intensities)
# %%
timestamps_one = np.load("../data/pl_230603/timestamps_16_53_12.npy")
timestamps_two = np.load("../data/pl_230603/timestamps_17_08_40.npy")
timestamps_two[:,0] += 1000
dm_timestamps = np.vstack((timestamps_one, timestamps_two))
dm_timestrings = [str(10000 * r[1] + 100 * r[2] + r[3]) for r in dm_timestamps]

inputzs_one = np.load("../data/pl_230603/inputzs_16_53_12.npy")
inputzs_two = np.load("../data/pl_230603/inputzs_17_08_40.npy")
inputzs = np.vstack((inputzs_one, inputzs_two))
# %%
def find_dm_iteration(stamp, last=0):
    itr = last
    if stamp < dm_timestrings[0] or stamp > dm_timestrings[-1]:
        return -1
    while stamp > dm_timestrings[itr+1]:
        itr += 1

    return itr
# %%
dm_iterations = []
i = 0
for s in np.sort(timestamps):
    i = find_dm_iteration(s, i)
    dm_iterations.append(i)
# %%
