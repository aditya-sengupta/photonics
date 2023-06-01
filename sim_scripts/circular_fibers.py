# %%
import numpy as np

from hcipy import *
from hcipy.mode_basis import zernike_ansi

from lightbeam.LPmodes import lpfield
from lightbeam.optics import ScaledCyl
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize

wl = 1.0 # um
ds = 0.25 # um
n = 1.444
r = 5.0 # um
z_ex = 10000.0 # um
PML = int(4 / ds)

mesh = RectMesh3D(
    xw = 128,
    yw = 128,
    zw = z_ex,
    ds = ds,
    dz = 10,
    PML = PML
)

cyl = ScaledCyl([0.0, 0.0], r, z_ex, n, nb=1)
cyl.set_sampling(mesh.xy)
# %%
prop = Prop3D(wl0 = wl, mesh = mesh, optical_system = cyl, n0 = n)
# %%
xg, yg = np.meshgrid(mesh.xy.xa[PML:-PML], mesh.xy.ya[PML:-PML], indexing='ij')
s = r / (xg[1,0] - xg[0,0]) + 1/2
pupil_grid = make_pupil_grid(mesh.xy.shape, 1)
focal_grid = make_focal_grid(11, s / 11)
# shaneAO f number
fprop = FraunhoferPropagator(pupil_grid, focal_grid)
w = mesh.xy.get_weights()
mask = (xg ** 2 + yg ** 2 <= r ** 2)
d = int((xg.shape[0] - 2 * s) // 2)
# %%
def plot_zernike_prop(zern, ampl):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0)
    u_in = np.array(fprop(wf).electric_field.shaped)
    d = int((xg.shape[0] - 2 * s) // 2)
    u_in = normalize(np.pad(u_in, ((d, d), (d, d))) * mask)
    u_out = prop.prop2end(u_in, remesh_every=0)
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(np.abs(u_in))
    axs[0].set_title(f"Zernike {zern} input intensity")
    axs[1].imshow(np.abs(u_out))
    axs[1].set_title("Zernike fiber output")


def plot_lp_prop(l, m):
    u_in = normalize(lpfield(xg, yg, l, m, r, wl, n, 1))
    u_out = prop.prop2end(u_in, remesh_every=0)
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(np.abs(u_in))
    axs[0].set_title(f"LP {l}, {m} input intensity")
    axs[1].imshow(np.abs(u_out))
    axs[1].set_title("LP fiber output")

# %%
plot_lp_prop(3, 2)
# %%
plot_zernike_prop(5, 1.0)

# %%
