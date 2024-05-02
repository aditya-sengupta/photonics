"""
This script attempts to find large changes in intensity over time in video data taken from Matt's camera using the ShaneAO calibration source on 2023-06-02. This was unsuccessful and this script will be deprecated when I've finished a writeup of the ShaneAO runs.
"""

# %%
import numpy as np
import os

from collections import Counter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

from photonics import PROJECT_ROOT
from photonics.lantern_reader import LanternReader

reader = LanternReader(
    nports = 18,
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

for subdir in ["randomcombinations_subset"]:# ["mode_combinations_small", "mode_combinations", "randomcombinations", "onsky4_part1"]:
    reader.subdir = "pl_230602/" + subdir
    dirname = os.path.join(PROJECT_ROOT, "data", reader.subdir)
    fnames = np.sort([x for x in os.listdir(dirname) if x.endswith("." + reader.ext)])
    diff_intensities = np.zeros((len(fnames), reader.nports))
    last_image = np.zeros_like(img)
    for (i, fname) in enumerate(tqdm(fnames)):
        img = reader.read_image(os.path.join(dirname, fname[:-5]))
        diff_intensities[i,:] = reader.get_intensities(img - last_image)
        last_image = img

np.save(os.path.join(dirname, f"{subdir}_diff_intensities.npy"), diff_intensities)

peaks = Counter(
    np.hstack([
        find_peaks(diff_intensities[:,i], distance=3)[0]
        for i in range(reader.nports)
    ])
)
peaks = np.array(list(filter(lambda x: peaks[x] > 5, peaks)))[:20] * 0.1
plt.stem(np.sort(peaks), np.ones_like(peaks))

overall_peaks = find_peaks(np.sum(diff_intensities, axis=1), distance=10)[0][:20] * 0.1
plt.stem(overall_peaks, np.ones_like(overall_peaks))