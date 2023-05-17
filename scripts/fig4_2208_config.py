# %%
import os, subprocess, sys
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

from itertools import product

from hcipy import *
from hcipy.mode_basis import zernike_ansi

from lightbeam.LPmodes import lpfield
from lightbeam.optics import make_lant6_saval, make_lant3big
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize, norm_nonu, overlap_nonu, chain, chain_apply, lmc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR.endswith("scripts"):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
# %%
D = 3.0 # m
wl = 1.55 # um
ds = 1/2
nclad = 1.444
ncore = nclad + 0.0088
njack = nclad - 5.5e-3
rclad = 10
scale = 8
rcore = 2.2 / scale 
offset0 = rclad * 2/3
PML = int(4 / ds)

mesh = RectMesh3D(
    xw = 256,
    yw = 256,
    zw = 10000,
    ds = ds,
    dz = 10,
    PML = PML
)

lant = make_lant6_saval(offset0, rcore, rclad, 0, mesh.zw, (ncore, nclad, njack), final_scale=scale)

# %%
lant.set_sampling(mesh.xy)
prop = Prop3D(wl0 = wl, mesh = mesh, optical_system = lant, n0 = nclad)

xg, yg = np.meshgrid(mesh.xy.xa[PML:-PML], mesh.xy.ya[PML:-PML], indexing='ij')
s = rclad / (xg[1,0] - xg[0,0]) + 1/2
pupil_grid = make_pupil_grid(mesh.xy.shape, D)
focal_grid = make_focal_grid(11, s / 11, spatial_resolution=(wl * 1e-6 / D))
# shaneAO f number
fprop = FraunhoferPropagator(pupil_grid, focal_grid)
w = mesh.xy.get_weights()
mask = (xg ** 2 + yg ** 2 <= rclad ** 2)
d = int((xg.shape[0] - 2 * s) // 2)

# %%
def save_for_zampl(zern, ampl, save=True):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0)
    u_inz = np.array(fprop(wf).electric_field.shaped)
    u_inz = normalize(np.pad(u_inz, ((d, d), (d, d))) * mask)
    u = prop.prop2end(u_inz, remesh_every=0)
    if save:
        np.save(os.path.join(ROOT_DIR, f"data/zerns/2208_4_{zern}_{ampl}.npy"), u)
    return u

assert lant.check_smfs(2 * np.pi / wl)
assert lant.check_mode_support(2 * np.pi / wl)

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
    axs[0].imshow(np.array(phase.shaped))
    axs[0].set_title(f"Zernike {zern} input phase")
    axs[1].imshow(np.abs(u_out))
    axs[1].set_title("Zernike lantern output")

# %%
def plot_lp_prop(l, m):
    u_in = normalize(lpfield(xg, yg, l, m, rclad, wl, ncore, nclad))
    u_out = prop.prop2end(u_in, remesh_every=0)
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(np.abs(u_in))
    axs[0].set_title(f"LP {l}, {m} input intensity")
    axs[1].imshow(np.abs(u_out))
    axs[1].set_title("LP lantern output")

# %%
if __name__ == "__main__":
    plot_zernike_prop(5, 1.0)
    # %%
    plot_lp_prop(3, 2)
    # %%
    u_outz = save_for_zampl(1, 1.0) # vary 1-5 and -1.0 to 1.0
    output_powers = []
    for pos in lant.final_core_locs:
        _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
        output_powers.append(np.power(overlap_nonu(_m,u_outz,w),2))

    print(output_powers)
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')
    if len(sys.argv) > 1:
        zerns = [int(sys.argv[1])]
    else:
        zerns = [1, 2, 3, 4, 5]

    if len(sys.argv) > 2:
        ampls = [float(sys.argv[2])]
    else:
        ampls = np.linspace(-1, 1, 11)
    
    for (zern, ampl) in product(zerns, ampls):
        save_for_zampl(zern, ampl)

# %%
