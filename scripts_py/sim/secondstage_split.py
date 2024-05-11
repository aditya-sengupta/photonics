# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
from photonics.utils import lmap, rms
from photonics.optics import Optics
from photonics.pyramid_optics import PyramidOptics
from photonics.lantern_optics import LanternOptics
from photonics.second_stage import correction
# %%
n_filter = 9
f_cutoff = 30
f_loop = 100
dt = 1 / f_loop
lantern_fnumber = 6.5
dm_basis = "modal"
optics = Optics(lantern_fnumber, dm_basis)
pyramid = PyramidOptics(optics, dm_basis)
lantern = LanternOptics(optics, lantern_fnumber)
optics.turbulence_setup(fried_parameter=0.1, seed=10)

# %%
lantern.show_linearity(*lantern.make_linearity(optics))
# %%
pyramid_correction = correction(optics, pyramid, lantern, use_lantern=False)
# %%
lo.readout(pyramid_correction["wavefronts_after_dm"][0])
# %%
sso.deformable_mirror.actuators[2] = 0.25 / sso.wf.wavenumber
wf_after_dm = sso.deformable_mirror.forward(sso.wf)
focal_image = sso.focal_propagator(wf_after_dm)
fig, axs = plt.subplots(1, 3, figsize=(10,5))
imshow_field(sso.deformable_mirror.surface / 1e-6, ax=axs[2])
imshow_field(np.log10(focal_image.intensity / np.max(focal_image.intensity)), ax=axs[0])
axs[1].imshow(np.abs(lo.lantern_output_to_plot(focal_image)) ** 2)
plt.colorbar()
# %%
sso.deformable_mirror.flatten()
# %%
# %%
sso.deformable_mirror.actuators[2] = sso.wl * 0.25 / (2 * np.pi)
wf_after_dm = sso.deformable_mirror.forward(sso.wf)
focal_image = sso.focal_propagator(wf_after_dm)
imshow_psf(focal_image.intensity)
plt.colorbar()
sso.deformable_mirror.flatten()
focal_ref_wf = sso.focal_propagator(sso.deformable_mirror.forward(sso.wf))
# %%
plt.plot(np.abs(lo.lantern_coeffs(lo.zernike_to_focal(4, 0.5))))
plt.plot(np.abs(lo.lantern_coeffs(focal_image)))
# %%
hc.get_strehl_from_focal(focal_image.intensity/sso.norm,focal_ref_wf.intensity/sso.norm)
# %%
hc.get_strehl_from_focal(lo.zernike_to_focal(4, 0.5).intensity/sso.norm,sso.lantern_optics.zernike_to_focal(4, 0.0).intensity/sso.norm)

# %%
