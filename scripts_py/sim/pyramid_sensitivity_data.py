# %%
from photonics.utils import rms
# %%
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.lantern_optics import LanternOptics
import numpy as np
# %%
optics = Optics()
pyr = PyramidOptics(optics, modsteps=4)
# %%

optics.deformable_mirror.actuators[0] = 9 * 1.55e-8
wf_after_dm = optics.deformable_mirror.forward(optics.wf)
rms(wf_after_dm.phase)
# %%
np.save("../../data/sensitivity_test/aperture_efield.npy", optics.wf.electric_field)
np.save("../../data/sensitivity_test/pyramid_flat.npy", pyr.readout(optics.wf))
np.save("../../data/sensitivity_test/pyramid_im.npy", pyr.interaction_matrix)

# %%
lo = LanternOptics(optics, nmodes=19)
# %%
np.save("../../data/sensitivity_test/lantern_flat.npy", lo.readout(optics.wf))
np.save("../../data/sensitivity_test/lantern_im.npy", lo.interaction_matrix)
# %%
