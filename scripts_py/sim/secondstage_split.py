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
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.second_stage import correction

# %%
n_filter = 9
f_cutoff = 30
f_loop = 800
dt = 1/f_loop
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
lantern = LanternOptics(optics)
optics.turbulence_setup(fried_parameter=0.1, seed=1)
corr = partial(correction, optics=optics, pyramid=pyramid, lantern=lantern, f_loop=f_loop)
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
