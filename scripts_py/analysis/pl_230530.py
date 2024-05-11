# %%
import numpy as np
from os import path
from tqdm import tqdm

from photonics.experiments.lantern_reader import LanternReader

reader = LanternReader(
    nports = 18,
    fwhm = 18,
    ext = "npy",
    threshold = 100,
    imgshape = (1200, 1920),
    subdir = "pl_230530"
)

intensities_path = reader.filepath("pl_intensities_230530_1504")
if path.isfile(intensities_path):
    all_intensities = np.load(intensities_path)
else:
    imgs = reader.read_image("pls_230530_1504")
    reader.set_centroids(imgs[0])
    all_intensities = np.zeros((len(imgs), reader.nports))
    for (i, img) in enumerate(tqdm(imgs)):
        all_intensities[i] = reader.get_intensities(img)

    np.save(reader.filepath("pl_intensities_230530_1504"), all_intensities)
# %%
