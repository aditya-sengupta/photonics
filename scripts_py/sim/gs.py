# %%
from IPython import get_ipython
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.optics import Optics
from hcipy import imshow_field
from photonics.utils import nanify
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# %%
optics = Optics(lantern_fnumber=6.5)
lo = LanternOptics(optics)
# %%
lo.show_GS(optics, 3, 0.5, niter=10)
# %%
gs_ret = lo.GS(
    optics,
    lo.forward(
        optics,
        optics.zernike_to_pupil(3, 0.1)
    ).intensity,
    niter=20
)
# %%
optics.zernike_basis.coefficients_for(optics.zernike_to_pupil(3, 0.1).phase)[:lo.nmodes]
# %%
optics.zernike_basis.coefficients_for(gs_ret.phase)[:lo.nmodes]
# %%
imshow_field(nanify(gs_ret.phase - np.mean(gs_ret.phase), optics.aperture), cmap="RdBu")
plt.gca().set_axis_off()
# %%
z_applied = 2
a_applied = -0.5
zdecomps = []
EM_in, measured_in, measured_out = lo.GS_init(
    optics,
    lo.forward(
        optics,
        optics.zernike_to_pupil(z_applied, a_applied)
    ).intensity
)
zdecomps.append(
    optics.zernike_basis.coefficients_for(EM_in.phase)[:lo.nmodes]
)
for i in trange(4):
    EM_in = lo.GS_iteration(optics, EM_in, measured_in, measured_out)
    zdecomps.append(
        optics.zernike_basis.coefficients_for(EM_in.phase)[:lo.nmodes]
    )

zdecomps = np.array(zdecomps).T

# %%
plt.hlines(a_applied, 0, zdecomps.shape[1] - 1, label="Target")
for (i, r) in enumerate(zdecomps):
    if i == z_applied:
        plt.plot(r, label="Injected mode", color="k")
    else:
        plt.plot(r, alpha=0.1, color="r")
plt.xticks(np.arange(0, zdecomps.shape[1], 4))
plt.xlabel("Gerchberg-Saxton iteration")
plt.ylabel("Amplitude (rad)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.subplots_adjust(right=0.8)

# %%
