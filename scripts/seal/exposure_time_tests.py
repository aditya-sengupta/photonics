import numpy as np
from time import sleep

def max_counts_per_aperture(reader, img):
    r = reader.fwhm
    max_vals = []
    for (x, y) in zip(reader.xc, reader.yc):
        lower_x, upper_x = int(np.floor(x - r)), int(np.ceil(x + r))
        lower_y, upper_y = int(np.floor(y - r)), int(np.ceil(y + r))
        masked = img[
            lower_y:upper_y,
            lower_x:upper_x
        ]
        max_vals.append(np.max(masked))

    return max_vals

exposure_times = np.arange(5, 51)
darks = []
for t in exposure_times:
    c.set_exp(t)
    darks.append(c.get() + c.dark)

input("flip laser")
maxes = []
for (t, dark) in zip(exposure_times, darks):
    c.set_exp(t)
    maxes.append(max_counts_per_aperture(reader, c.get() + c.dark - dark))

maxes = np.array(maxes)
saturation_threshold = 60_000

plt.plot(exposure_times, np.sum(maxes > saturation_threshold, axis=1))
plt.xlabel("Exposure time (ms)")
plt.ylabel("Number of saturated ports")
