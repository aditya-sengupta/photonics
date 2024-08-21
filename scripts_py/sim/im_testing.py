# %%
import numpy as np
from matplotlib import pyplot as plt
# %%
from photonics.simulations.optics import Optics
from photonics.simulations.lantern_optics import LanternOptics
# %%
from photonics.simulations.command_matrix import make_command_matrix
# %%
optics = Optics()
lo = LanternOptics(optics)
# %%
make_command_matrix(optics.deformable_mirror, lo, optics.wf)
# %%
dm = optics.deformable_mirror
optics.deformable_mirror.flatten()
# %%
dm.actuators[2] = -0.01 / (4 * np.pi / 1.55e-6)
flat_reading = lo.readout(optics.wf)
wf = dm.forward(optics.wf)
lo_reading_minus = lo.readout(wf) - flat_reading
dm.actuators[2] = +0.01 / (4 * np.pi / 1.55e-6)
wf = dm.forward(optics.wf)
lo_reading_plus = lo.readout(wf) - flat_reading
plt.plot(lo_reading_plus)
plt.plot(lo_reading_minus)
# %%
lo.show_linearity(*lo.make_linearity(optics))
# %%
