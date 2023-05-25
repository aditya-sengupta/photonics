
# %%
from skimage.restoration import unwrap_phase
from hcipy import *
from hcipy.optics import DeformableMirror
from lightbeam.LPmodes import get_V, get_modes, get_num_modes, make_complex_lp_basis
from functools import reduce

from fig4_2208_config import *

wl = 0.65e-6 # m
 # %%
zoomed_focal_grid = make_focal_grid(len(xg) / (4 * num_airy), num_airy, spatial_resolution=spatial_resolution * rclad / focal_grid.x[-1])

V = get_V(2 * np.pi / wl, rclad, ncore, nclad)
assert 0.9 < V < 9e6
lp_basis = make_complex_lp_basis(zoomed_focal_grid, rclad, wl, ncore, nclad)
N = len(lp_basis)
assert 1 < N < 100
lp_pupil_basis = ModeBasis(lmc(
        lp_basis,
        lambda lp: Wavefront(lp, wl),
        lambda wf: fprop.backward(wf),
        lambda wf: wf.electric_field
    ), pupil_grid)
dm = DeformableMirror(lp_pupil_basis)
aperture = evaluate_supersampled(make_circular_aperture(D), pupil_grid, 4)
wf = Wavefront(aperture, wl)

# %%
imshow_field(lp_basis[-5]); plt.colorbar()
# %%
if False:
    dm.flatten()
    dm.actuators[4] = 1
    wfi = fprop(dm.forward(wf)) # fprop
    imshow_field(wf.power, cmap='inferno')
    plt.colorbar()

# 1920 * 1200 at the lantern output, each one is (5.86 um)^2
# %%
if False:
    amp = 0.1
    from functools import reduce
    N = 6
    patterns = amp * np.array(
        reduce(lambda x, y: x + y, [
            [[float(i == j or i == j + k) for i in range(N)] for j in range(N - k)] for k in range(3)
        ])
    )
    fig, ax = plt.subplots()
    ax.imshow(np.flipud(patterns.T), extent=[0, len(patterns), 0.5, N+0.5])
    ax.set_xticks([])
    ax.set_yticks(np.arange(N)+1)
    ax.set_xlabel("Queries")
    ax.set_ylabel("Basis mode combinations")
    ax.set_title(f"ID procedure for a {N}-port photonic lantern propagation matrix")
    plt.savefig("../figures/id_procedure.png", bbox_inches='tight')

# %%
modes = get_modes(V)
nr = 5
fig, axs = plt.subplots(nr, int(np.ceil(len(lp_pupil_basis) / nr)), )
i = 0
for axr in axs:
    for ax in axr:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axr:
        l, m = modes[i]
        phase_screen = np.angle(lp_pupil_basis[i]).shaped
        # ax.imshow(unwrap_phase(phase_screen))
        phase_unwrapped = Field(unwrap_phase(phase_screen).ravel(), pupil_grid)
        imshow_field(phase_unwrapped, mask=aperture.astype(bool), ax=ax)
        ax.set_title(f"{l}, {m}", fontsize=6)
        i += 1
        if i == len(lp_pupil_basis):
            break
plt.savefig("../figures/lp_pupils.png", bbox_inches='tight')

# %%
