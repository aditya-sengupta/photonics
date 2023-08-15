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

from joblib import Parallel, delayed

from tqdm import trange, tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR.endswith("sim"):
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
# %%
D = 3.0 # m
wl = 1.55 # um
ds = 1/4
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
lant.set_sampling(mesh.xy)

# %%
def binarize(a):
    return np.divide(a, a, out=np.zeros_like(a), where=a!=0)
# %%
out = np.zeros(mesh.xy.shape)
lant.set_IORsq(out,0)
#plt.imshow(out,vmin=njack*njack,vmax=ncore*ncore)
#plt.show()

# %%
prop = Prop3D(wl0 = wl, mesh = mesh, optical_system = lant, n0 = nclad)

f = 51.0
xg, yg = np.meshgrid(mesh.xy.xa[PML:-PML], mesh.xy.ya[PML:-PML], indexing='ij')
s = rclad / (xg[1,0] - xg[0,0]) + 1/2
pupil_grid = make_pupil_grid(mesh.xy.shape, D)
focal_grid = make_focal_grid(11, s / 11, spatial_resolution=(wl * 1e-6 * f / D))
# shaneAO f number
fprop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=f)
w = mesh.xy.get_weights()
mask = out[PML:-PML, PML:-PML] > njack*njack
d = int((xg.shape[0] - 2 * s) // 2)

ref_phase = zernike_ansi(1)(pupil_grid)
ref_aberration = np.exp(1j * ref_phase)
ref_wf = Wavefront(ref_aberration, wavelength=prop.wl0)
ref_u = np.array(fprop(ref_wf).electric_field.shaped)
ref_u = normalize(np.pad(ref_u, ((d, d), (d, d))) * mask)
out_core = (out[PML:-PML, PML:-PML] > njack*njack).astype(np.float64)
plt.imshow(np.logical_xor(binarize(out_core), binarize(ref_u)))
plt.title("XOR of FMF input and beam input")
# %%
def save_for_zampl(zern, ampl, save=True, verbose=True, r=trange):
    if verbose:
        print(zern, ampl, "\n")
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0)
    u_inz = np.array(fprop(wf).electric_field.shaped)
    u_inz = normalize(np.pad(u_inz, ((d, d), (d, d))) * mask)
    u = prop.prop2end(u_inz, remesh_every=0, verbose=False, r=r)
    savepath = os.path.join(ROOT_DIR, f"data/zerns/2208_4_{zern}_{ampl}.npy")
    if save:
        np.save(savepath, u)
        if verbose:
            print(savepath)
    return u

assert lant.check_smfs(2 * np.pi / wl)
assert lant.check_mode_support(2 * np.pi / wl)

# %%
def plot_zernike_prop(zern, ampl):
    phase = zernike_ansi(zern)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0*1e-6)
    u_in = np.array(fprop(wf).electric_field.shaped)
    d = int((xg.shape[0] - 2 * s) // 2)
    u_in = normalize(np.pad(u_in, ((d, d), (d, d))) * mask)
    u_out = prop.prop2end(u_in, remesh_every=0)
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(np.angle(u_in))
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
#plot_zernike_prop(3, 0.6)

# %%
#plot_zernike_prop(5, 1.0)
# %%
#u_outz = save_for_zampl(1, 1.0) # vary 1-5 and -1.0 to 1.0
"""output_powers = []
for pos in lant.final_core_locs:
    _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
    output_powers.append(np.power(overlap_nonu(_m,u_outz,w),2))
print("output_powers")"""

def prop_and_save(u_inz, z, a):
    print(z, a)
    u = prop.prop2end(u_inz, remesh_every=0, verbose=False, r=range)
    savepath = os.path.join(ROOT_DIR, f"data/zerns/2208_4_{z}_{a}.npy")
    print(savepath)
    np.save(savepath, u)

# %%
if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    zerns = [1, 2]
    ampls = np.linspace(-1, 1, 5)

    input_fields = []
    for (zern, ampl) in tqdm(list(product(zerns, ampls))):
        phase = zernike_ansi(zern)(pupil_grid)
        aberration = np.exp(1j * ampl * phase)
        wf = Wavefront(aberration, wavelength=prop.wl0)
        u_inz = np.array(fprop(wf).electric_field.shaped)
        u_inz = normalize(np.pad(u_inz, ((d, d), (d, d))) * mask)
        input_fields.append(u_inz)

    # %%
    Parallel(n_jobs=10)(delayed(prop_and_save)(u, z, a) for u, (z, a) in zip(input_fields, product(zerns, ampls)))

# %%
