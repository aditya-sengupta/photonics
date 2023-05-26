# %%
import sys

from itertools import product

from lightbeam.LPmodes import lpfield

sys.path.append("..")
from src.lantern_reader import LanternReader

reader = LanternReader(
    nports = 18,
    fwhm = 15,
    ext = "png",
    imgshape = (1200, 1920)
)

guess_centroids = np.array([
    [117, 782],
    [178, 993],
    [304, 1171],
    [334, 715],
    [391, 931],
    [440, 408],
    [508, 1239],
    [529, 608],
    [582, 824],
    [586, 1037],
    [655, 401],
    [771, 918],
    [844, 504],
    [902, 1095],
    [990, 895],
    [255, 510],
    [770, 710],
    [970, 680]
])

# %%
def approx_peak_locs(img, num_peaks=18, spot_size=25, min_energy=10_000):
    imgc = img.copy()
    locs = []
    i = 0
    while i < num_peaks and not(np.all(imgc == 0)):
        ind = np.unravel_index(np.argmax(imgc, axis=None), img.shape)
        if np.sum(imgc[ind[0]-spot_size:ind[0]+spot_size, ind[1]-spot_size:ind[1]+spot_size]) > min_energy:
            locs.append(ind)
            imgc[ind[0]-spot_size:ind[0]+spot_size, ind[1]-spot_size:ind[1]+spot_size] = 0
            i += 1

    return locs

def refine_peak_locs(img, peaks, cutout_size=25):
    for (xc, yc) in peaks:
        img[xc-cutout_size:xc+cutout_size,]
    return peaks
# %%
img = reader.read_image("../data/pl_flat_230525_1326")
peaks = approx_peak_locs(img)

# %%
plt.imshow(img, cmap='magma')
# %%
