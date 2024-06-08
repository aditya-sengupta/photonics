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
import matplotlib.cm as cm
from photonics.utils import lmap, rms, zernike_names
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.second_stage import correction

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# %%
n_filter = 9
f_cutoff = 30
f_loop = 800
dt = 1/f_loop
optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
lantern = LanternOptics(optics)
corr = partial(correction, optics=optics, pyramid=pyramid, lantern=lantern, f_loop=f_loop, f_cutoff=f_cutoff)
focus_ncpa = optics.zernike_to_pupil(2, 0.3)

# %%
niter = 800
second_stage_iter = 200
D_over_r0s = [1, 2, 4, 8, 16, 32, 64]
colors = cm.cool(np.linspace(0, 1, len(D_over_r0s)))
for (c, D_over_r0) in zip(colors, D_over_r0s):
    optics.turbulence_setup(fried_parameter=optics.telescope_diameter/D_over_r0, seed=371)
    twostage_correction = corr(use_pyramid=True, use_lantern=True, num_iterations=niter, ncpa=focus_ncpa, perfect_pyramid=False, perfect_lantern=False, second_stage_iter=second_stage_iter)
    plt.plot(twostage_correction["time"], twostage_correction["strehl_ratios"], label=f"$D/r_0$ = {D_over_r0}", color=c)

plt.vlines([second_stage_iter / 800], 0, 1, "k", "dashed", label="PL loop closed")
plt.xlabel("Time (s)")
plt.ylabel("Strehl ratio")
plt.title("Multi-WFS correction with linear PL reconstruction")
plt.legend(bbox_to_anchor=(1.0,1.0))
# %%
