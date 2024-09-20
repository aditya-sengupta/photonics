# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from photonics.experiments.lantern_reader import LanternReader
# %%
reader = LanternReader(
    nports = 18, 
    fwhm = 18,
    threshold = 25,
    ext = "npy",
    imgshape = (1200, 1920),
    subdir = "pl_231018"
)
# %%
times, averages, peaks, xc, yc = map(lambda x: np.load(reader.filepath(x)), ["exptimes", "expavgs", "exppeaks", "xc", "yc"])
reader.xc = xc
reader.yc = yc
pixels_per_port = np.array([np.sum((reader.xi - xv) ** 2 + (reader.yi - yv) ** 2 <= reader.fwhm ** 2) for (xv, yv) in zip(xc, yc)])
# %%
plt.plot(times, peaks)
saturation_indices = np.argmax(peaks, axis=0)
saturation_times = times[saturation_indices]
# %%
darks_intensities = np.load(reader.filepath("dark_intensities"))
# %%
plt.plot(times[1:], averages[1:])
linear_fits = [linregress(times[1:saturation_indices[i]], (averages + dark_intensities)[1:saturation_indices[i],i]) for i in range(18)]
_, read_noise = list(map(lambda x: x.slope, linear_fits)), list(map(lambda x: x.intercept, linear_fits))
# %%
signal = averages[1:]
noise = np.sqrt(signal + dark_intensities[1:] + np.array(read_noise))
snr = signal/noise
for i in range(18):
    plt.plot(times[1:saturation_indices[i]], snr[:saturation_indices[i]-1,i], c=plt.cm.winter(i / reader.nports))
plt.xlabel("Exposure time (ms)")
plt.ylabel("Signal-to-noise ratio")
plt.title("SNR per lantern port, cut off at saturation, 0.16V, 18 frames")
# %%
