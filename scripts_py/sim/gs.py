# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.optics import Optics
from hcipy import imshow_field
from photonics.utils import nanify, zernike_names
from photonics.linearity import plot_linearity
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
# %%
optics = Optics(lantern_fnumber=6.5)
lo = LanternOptics(optics)
# %%
z, a = 16, -0.5
input_phase = optics.zernike_to_phase(z, a)
input_pupil = optics.phase_to_pupil(input_phase)
guess_pupil = optics.phase_to_pupil(optics.zernike_to_phase(z, a))
lo.show_GS(optics, z, a, guess=guess_pupil, niter=200)
# %%
gs_ret = lo.GS(
    optics,
    lo.forward(
        optics,
        optics.zernike_to_pupil(3, 0.1)
    ).intensity,
    niter=4
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
zr = np.arange(9)
ar = np.arange(-1.0, 1.01, 0.01)
sweep = np.zeros((len(zr), len(ar), len(zr)))

# %%
for (i, z) in enumerate(zr):
    print(zernike_names[i])
    for (j, a) in enumerate(tqdm(ar)):
        EM_in = lo.GS(
            optics,
            lo.forward(
                optics,
                optics.zernike_to_pupil(z, a)
            ).intensity
        )
        retrieved_zernikes = optics.zernike_basis.coefficients_for(EM_in.phase)[:len(zr)]
        sweep[i,j,:] = retrieved_zernikes
        
# %%
np.save("../../data/linear_sweeps/gs.npy", sweep)
# %%
plot_linearity(ar, sweep, "Gerchberg-Saxton")
# %%
# Gerchberg Saxton error reduction