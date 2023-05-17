
# %%
from hcipy import *
from hcipy.optics import DeformableMirror
from lightbeam.LPmodes import get_V, get_num_modes
from juliacall import Main as jl

from fig4_2208_config import *
 # %%
V = get_V(2 * np.pi / wl, rclad, ncore, nclad)
assert 0.9 < V < 9e6
lp_basis = make_lp_modes(focal_grid, V, 1e-6)
N = get_num_modes(2 * np.pi / wl, rclad, ncore, nclad)
assert 1 < N < 50
lp_pupil_basis = ModeBasis(lmc(
        lp_basis,
        lambda lp: Wavefront(lp, wl),
        lambda wf: fprop.backward(wf),
        lambda wf: wf.electric_field
    ), pupil_grid)
dm = DeformableMirror(lp_pupil_basis)
aperture = evaluate_supersampled(make_circular_aperture(D), pupil_grid, 4)
wf = Wavefront(aperture, wl)

imshow_field(lp_basis[6])

# %%
imshow_field(lp_pupil_basis[6])
# %%
dm.flatten()
dm.actuators[2] = 1e16
wfi = fprop(dm.forward(wf)) # fprop
imshow_field(np.log10(wf.power / wf.power.max()), cmap='inferno')
plt.colorbar()

# 1920 * 1200 at the lantern output, each one is (5.86 um)^2
# %%
patterns = amp * np.array(
    reduce(lambda x, y: x + y, [
        [[float(i == j or i == j + k) for i in range(N)] for j in range(N - k)] for k in range(3)
    ])
)
# plt.imshow(patterns)
# %%
