# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
from photonics.utils import lmap, rms
from photonics.second_stage_optics import SecondStageOptics
# %%
sso = SecondStageOptics(f_loop=800, dm_basis="modal", ncpa_z=2, ncpa_a=0.0)
sso.turbulence_setup(fried_parameter=0.1, seed=10)
# %%
lo = sso.lantern_optics
# %%
pyramid_correction = sso.correction(num_iterations=200, two_stage_niter=201)
# %%
lantern_correction = lo.correction(sso.layer, sso.deformable_mirror)
# %%
lo.readout(pyramid_correction["wavefronts_after_dm"][0])
# %%
sso.deformable_mirror.actuators[2] = sso.wl * 0.1
wf_after_dm = sso.deformable_mirror.forward(sso.wf)
focal_image = sso.focal_propagator(wf_after_dm)
fig, axs = plt.subplots(1, 2, figsize=(10,5))
imshow_field(np.log10(focal_image.intensity / np.max(focal_image.intensity)), ax=axs[0])
plt.imshow(np.abs(lo.lantern_output_to_plot(focal_image)) ** 2)
sso.deformable_mirror.flatten()
# %%
