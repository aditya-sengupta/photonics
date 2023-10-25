import numpy as np
from time import sleep
from photonics import max_counts_per_aperture

exposure_times = np.arange(5, 51)
darks = []
for t in exposure_times:
    c.set_exp(t)
    darks.append(c.get() + c.dark)

input("flip laser")
maxes = []
snrs = []
for (t, dark) in zip(exposure_times, darks):
    c.set_exp(t)
    c.dark = dark
    maxes.append(max_counts_per_aperture(reader, c.get() + c.dark - dark))
    snrs.append(snr_per_port(plwfs))

maxes = np.array(maxes)
snrs = np.array(snrs)
saturation_threshold = 60_000

plt.plot(exposure_times, np.sum(maxes > saturation_threshold, axis=1))
plt.xlabel("Exposure time (ms)")
plt.ylabel("Number of saturated ports")

saturation_indices = np.argmax(maxes, axis=0)
for i in range(18):
    plt.plot(exposure_times[:saturation_indices[i]], snrs[:saturation_indices[i], i], c=plt.cm.winter(i/reader.nports))
plt.xlabel("Exposure time (ms)")
plt.ylabel("Signal-to-noise ratio")
plt.title("SNR per lantern port, cut off at saturation, 0.16mW, 9 frames")
plt.savefig(reader.filepath("snr"))