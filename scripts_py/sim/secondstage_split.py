# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import hcipy as hc
from matplotlib import pyplot as plt
from photonics.utils import lmap
from photonics.second_stage_optics import SecondStageOptics
# %%
nzern = 9
f_cutoff = 30 # Hz
sso = SecondStageOptics(f_loop=800)
sso.turbulence_setup(fried_parameter=0.5)
zernike_basis = hc.mode_basis.make_zernike_basis(nzern, sso.telescope_diameter, sso.pupil_grid, starting_mode=2)
nstep = 200
# %%
correction_results = sso.pyramid_correction(num_iterations=nstep, do_filter=False)
np.mean(correction_results["strehl_ratios"])
# %%
focal_plane_coeffs_over_time = np.array([np.array(zernike_basis.coefficients_for(x.phase)) for x in correction_results["wavefronts_after_dm"]])
# %%
pyramid_readings_in_rad = np.array(correction_results["pyramid_readings"])[:,:9] / (1.55e-6 / (2 * np.pi))

