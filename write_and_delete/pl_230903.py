"""
This script is deprecated as I don't think it will have any scientific use.
"""

# %%
import os
import photonics 
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm, trange
# %%
reader = photonics.LanternReader(
    nports = 18, # number of ports
    fwhm = 18, # the full width at half max of the lantern spots; feeds into DAOStarFinder
    threshold = 25, # threshold number of SDs to be above the background; feeds into DAOStarFinder
    ext = "npy", # the kind of file we want to write to
    imgshape = (1200, 1920), # the shape of the camera images we're writing
    subdir = "pl_shane_230903" # the subdirectory of "data" we write to; defaults to "pl_YYMMDD".
)
# %%
reader.set_centroids(iio.imread(reader.filepath("0-flat", ext="png")))
# %%
N = 12
zernikes = np.zeros((2 * N, N))
intensities = []
for i, (n, sign) in enumerate(tqdm(product(range(1, N + 1), ["p", "n"]))):
    zernikes[i][n - 1] = 1 if sign == 'p' else -1
    img = iio.imread(reader.filepath(f"{n}-{sign}", ext="png"))
    intensities.append(reader.get_intensities(img))

intensities = np.vstack(intensities)
# %%
reader.save("interaction_zernikes", zernikes)
reader.save("interaction_intensities", intensities)
# %%
