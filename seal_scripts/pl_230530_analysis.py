# %%

import sys
sys.path.append("..")

from os import path
from tqdm import tqdm

from src.lantern_reader import LanternReader
# %%
reader = LanternReader(
    nports = 18,
    cutout_size = 20,
    fwhm = 18,
    ext = "npy",
    imgshape = (1200, 1920),
    subdir = "pl_230530"
)

intensities_path = reader.filepath("pl_intensities_230530_1514")
if path.isfile(intensities_path):
    all_intensities = np.load(intensities_path)
else:
    imgs = reader.read_image("pls_230530_1514")
    reader.set_centroids(imgs[0])
    all_intensities = np.zeros((len(imgs), reader.nports))
    for (i, img) in enumerate(tqdm(imgs)):
        all_intensities[i] = reader.get_intensities(img)

    np.save(reader.filepath("pl_intensities_230530_1514"), all_intensities)
# %%
