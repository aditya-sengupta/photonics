# %%
from photonics.lantern_optics import LanternOptics

lo = LanternOptics(f_number=10)
# %%
lo.show_GS(5, -0.5, niter=10)
# %%
