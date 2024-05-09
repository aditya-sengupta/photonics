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
nstep = 200
sso = SecondStageOptics(f_loop=800, dm_basis="modal", ncpa_z=4, ncpa_a=-0.0)
sso.turbulence_setup(fried_parameter=0.1, seed=10)
zernike_basis = hc.mode_basis.make_zernike_basis(nzern, sso.telescope_diameter, sso.pupil_grid, starting_mode=2)
correction_results = sso.correction(num_iterations=nstep, two_stage=False)
np.mean(correction_results["strehl_ratios"])
# %%