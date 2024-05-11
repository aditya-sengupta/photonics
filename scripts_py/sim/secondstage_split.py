# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
from functools import partial
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
corr = partial(correction, optics=optics, pyramid=pyramid, lantern=lantern)
focus_ncpa = optics.zernike_to_pupil(2, 0.3)
# %%
open_loop = corr(use_lantern=False, use_pyramid=False, ncpa=focus_ncpa)
# %%
pyramid_correction = corr(use_pyramid=True)
# %%
pyramid_correction_with_ncpa = corr(use_pyramid=True, ncpa=focus_ncpa)
# %%
lantern_correction = corr(use_lantern=True)
# %%
lantern_correction_with_ncpa = corr(use_lantern=True, ncpa=focus_ncpa)
# %%
twostage_correction = corr(use_pyramid=True, use_lantern=True, ncpa=focus_ncpa)
# %%
