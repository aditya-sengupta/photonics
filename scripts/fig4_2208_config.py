# %%
import numpy as np
from matplotlib import pyplot as plt

from hcipy import *
from hcipy.mode_basis import zernike_ansi
from lightbeam.LPmodes import lpfield
from lightbeam.optics import lant6_saval, lant3big
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize, norm_nonu, overlap_nonu

# %%
nclad = 1.4504
ncore = nclad + 0.0088
rclad = 6
rcore = 1.2
offset0 = rclad * 2/3
scale = 3

mesh = RectMesh3D(
    xw = 64,
    yw = 64,
    zw = 1000,
    ds = 1/4,
    dz = 1,
    PML = 32
)

lant = lant6_saval(
    rcore0 = rcore, rcore1 = rcore, rcore2 = rcore, rcore3 = rcore,
    rclad = rclad,
    ncore = ncore,
    nclad = nclad, njack = nclad - 5.5e-3,
    offset0 = offset0,
    z_ex = mesh.zw, final_scale=scale
)

lant.set_sampling(mesh.xy)
prop = Prop3D(wl0 = 1.0, mesh = mesh, optical_system = lant, n0 = nclad)

pupil_grid = make_pupil_grid(mesh.xy.shape, max(mesh.yg[0]) / rclad)
focal_grid = make_focal_grid(8, 10, reference_wavelength=1e-6*prop.wl0, f_number=17) 
s = focal_grid.shape[0]
# shaneAO f number
fprop = FraunhoferPropagator(pupil_grid, focal_grid)
# lant.show_cross_section(mesh.zw)

# %%
def save_for_zampl(zern, ampl):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0)
    u_in = np.array(normalize(fprop(wf).electric_field))
    u_in = u_in.reshape((s,s))
    u = prop.prop2end_uniform(u_in)
    # np.save(f"data/2208_4_{zern}_{ampl}.npy", u)
    return u

# %%
u_out = save_for_zampl(4, -0.3) # vary 2-6 and -1.0 to 1.0
# %%
output_powers = []
t = 2 * np.pi / 5
w = mesh.xy.get_weights()
core_locs = np.array([[0.0,0.0]] + [[offset0 * scale * np.cos(i*t), offset0 * scale * np.sin(i*t)] for i in range(5)])
for pos in core_locs:
    _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
    output_powers.append(np.power(overlap_nonu(_m,u_out,w),2))

print(output_powers)
# %%
