# %%
import numpy as np
from matplotlib import pyplot as plt
import hcipy as hc
from hcipy import imshow_field
from photonics.lantern_optics import LanternOptics
# %%
lo = LanternOptics(coupling=0.5)
lo.propagate_backwards()
lo.load_outputs()
# %%
lo.show_GS(6, 0.5)
# %%
lo.make_intcmd()
# %%
amplitudes, linearity_responses = lo.make_linearity(nzern=9, lim=0.01)
# %%
lo.show_linearity(amplitudes, linearity_responses)
# %%
_, proj_tip, _ = lo.lantern_output(lo.zernike_to_focal(1, 1.0))
_, proj_tilt, _ = lo.lantern_output(lo.zernike_to_focal(2, 1.0))
# %%
plt.imshow(np.abs(proj_tip) ** 2)
# %%
plt.imshow(np.abs(proj_tilt) ** 2)
# %%
