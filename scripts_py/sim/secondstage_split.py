# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
from photonics.utils import lmap, rms
from photonics import Optics, PyramidOptics, LanternOptics, correction

# %%
n_filter = 9
f_cutoff = 30
f_loop = 100
dt = 1/f_loop
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
lantern = LanternOptics(optics)
optics.turbulence_setup(fried_parameter=0.5, seed=10)
# %%
pyramid_correction = correction(optics, pyramid, lantern, use_lantern=False)
# %%
lantern_correction = correction(optics, pyramid, lantern, use_pyramid=False)
# %%
twostage_correction = correction(optics, pyramid, lantern, ncpa=optics.zernike_to_pupil(2,1.0))
# %%
