# %%
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from hcipy import imshow_field
from photonics.utils import nanify, rms
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
res = corr(use_pyramid=True, num_iterations=200)
# %%
# let's suppose we start from the end state of the pyramid CL
cl_phase = res["wavefronts_after_dm"][-1].phase
imshow_field(nanify(cl_phase, optics.aperture))
# %%
plt.plot([float(rms(x.phase - np.mean(x.phase))) for x in res["wavefronts_after_dm"]])
# %%
