# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import hcipy as hc
from hcipy import imshow_field
from matplotlib import pyplot as plt
from photonics.utils import lmap
from photonics.second_stage_optics import SecondStageOptics
# %%
sso = SecondStageOptics(f_loop=800, dm_basis="modal", ncpa_z=2, ncpa_a=0.0)
sso.turbulence_setup(fried_parameter=0.5, seed=10)
correction_results = sso.correction()
plt.plot(correction_results["strehl_ratios"])
# %%