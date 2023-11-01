# %%
import numpy as np
import imageio.v3 as iio
import os, sys, re

import matplotlib.pyplot as plt
from astropy.io import fits
# %%
d = fits.getdata("../data/pl_230602/Telemetry--UCOLick_2023-06-02/Data_0137.fits")
# %%
xc = d[1:, :144]
yc = d[1:, :144]
# %%
np.max(yc)
 # %%

