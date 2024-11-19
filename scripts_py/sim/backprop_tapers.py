# %%
import numpy as np
import hcipy as hc
# %%
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.optics import Optics
# %%
optics = Optics()
lo = LanternOptics(optics)
# %%
for i in np.arange(30_000, 65_000, 5_000):
    outputs = np.load(f"../../data/backprop_tapers/backprop_taper_{i}.npy")
    outputs = np.array([lo.sanitize_output(x) for x in outputs])
    projector = np.linalg.inv(outputs @ outputs.T) @ outputs
    print(i)
    lo.outputs = outputs
    lo.projector = projector
    lo.plot_outputs()

# %%
outputs = np.load(f"../../data/backprop_tapers/vary_core_sizes_backwards_19_base.npy")
outputs = np.array([lo.sanitize_output(x) for x in outputs])
projector = np.linalg.inv(outputs @ outputs.T) @ outputs
print(i)
lo.outputs = outputs
lo.projector = projector
lo.plot_outputs()

# %%
