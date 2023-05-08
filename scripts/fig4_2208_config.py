# %%
import os
import numpy as np
from matplotlib import pyplot as plt

from hcipy import *
from hcipy.mode_basis import zernike_ansi
from lightbeam.LPmodes import lpfield
from lightbeam.optics import make_lant6_saval, make_lant3big
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize, norm_nonu, overlap_nonu

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR.endswith("scripts"):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
# %%
wl = 1.0
ds = 1/4
nclad = 1.4504
ncore = nclad + 0.0088
njack = nclad - 5.5e-3
rclad = 4
scale = 4
rcore = 2.2/scale
offset0 = rclad * 1/2
PML = int(4 / ds)

mesh = RectMesh3D(
    xw = 64,
    yw = 64,
    zw = 1000,
    ds = ds,
    dz = 1,
    PML = PML
)

lant = make_lant6_saval(offset0, rcore, rclad, 0, mesh.zw, (ncore, nclad, njack), final_scale=scale)
# %%
t = 2 * np.pi / 5
lant.set_sampling(mesh.xy)
prop = Prop3D(wl0 = wl, mesh = mesh, optical_system = lant, n0 = nclad)

q = 8
xg, yg = np.meshgrid(mesh.xy.xa[PML:-PML], mesh.xy.ya[PML:-PML], indexing='ij')
s = (xg.shape[0]) / (q * 2)
pupil_grid = make_pupil_grid(mesh.xy.shape, 8)
focal_grid = make_focal_grid(8, s, reference_wavelength=1e-6*prop.wl0, f_number=17) 
s = focal_grid.shape[0]
# shaneAO f number
fprop = FraunhoferPropagator(pupil_grid, focal_grid)
# lant.show_cross_section(mesh.zw)
phase = zernike_ansi(zern)(pupil_grid)
imshow_field(phase)

# %%
def save_for_zampl(zern, ampl, save=True):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase) # * np.fft.fft2(lpfield(xg, yg, 0, 1, rclad, wl, ncore, nclad)).ravel()
    wf = Wavefront(aberration, wavelength=prop.wl0)
    u_in = np.array(normalize(fprop(wf).electric_field))
    u_in = u_in.reshape((s,s))
    # u_in = normalize(u_in * lpfield(xg, yg, 0, 1, rclad, wl, ncore, nclad))
    u, _ = prop.prop2end(u_in, remesh_every=0)
    if save:
        np.save(os.path.join(ROOT_DIR, f"data/zerns/2208_4_{zern}_{ampl}.npy"), u)
    return u

# %%
plt.imshow(np.abs(u_out))
# %%
_m = sum(
    norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w) for pos in lant.final_core_locs
)

plt.imshow(np.abs(_m * u_out))
plt.show()
# %%
lant.set_sampling(mesh.xy)
lant.show_cross_section(0)

# %%
if __name__ == "__main__":
    u_out = save_for_zampl(2, 1.0) # vary 2-6 and -1.0 to 1.0
    output_powers = []
    w = mesh.xy.get_weights()
    for pos in lant.final_core_locs:
        _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
        output_powers.append(np.power(overlap_nonu(_m,u_out,w),2))

    print(output_powers)


# %%
