# %%
from photonics.simulations.lantern_optics import LanternOptics

lo = LanternOptics(f_number=10.0)
# %%
lo.show_GS(3, -0.5, niter=10)
# %%
