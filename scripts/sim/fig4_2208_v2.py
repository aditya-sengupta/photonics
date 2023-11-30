# %%
from hcipy import *
from hcipy.mode_basis import zernike_ansi

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from tqdm import trange

from lightbeam.LPmodes import lpfield
from lightbeam.optics import make_lant6_saval, make_lant3big
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize

wavelength = 1550.0e-9
telescope_diameter = 3.0
pupil_grid = make_pupil_grid(60, telescope_diameter)
aperture = evaluate_supersampled(make_circular_aperture(telescope_diameter), pupil_grid, 10)
coupling_fraction = 0.3 # unaberrated PSF radius / input core radius
core_radius = 1.5e-6 # m
cladding_radius = 8.0e-6 # m; matching Lightbeam, but for now just step-index
target_psf_radius = coupling_fraction * core_radius
spatial_resolution = wavelength / telescope_diameter # m/m = fraction
focal_grid = make_focal_grid(q=10, num_airy=16, spatial_resolution=spatial_resolution)
fprop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=2 * target_psf_radius / spatial_resolution)
norm = np.max(fprop(Wavefront(aperture, wavelength=wavelength)).intensity)
ds = 1/4 # np.diff(focal_grid.x)[0] * 1e6
scale = 8
nclad = 1.444
ncore = nclad + 0.0088
njack = nclad - 5.5e-3
rcore = 2.2 / scale
rclad = cladding_radius * 1e6
offset0 = rclad * 2/3
PML = int(4 / ds)
xyw = (rclad) * scale * 2

mesh = RectMesh3D(
    xw = xyw * 2,
    yw = xyw * 2,
    zw = 10000,
    ds = ds,
    dz = 10,
    PML = PML
)

lant = make_lant6_saval(offset0, rcore, rclad, 0, mesh.zw, (ncore, nclad, njack), final_scale=scale)
lant.set_sampling(mesh.xy)
prop = Prop3D(wl0 = wavelength * 1e6, mesh = mesh, optical_system = lant, n0 = nclad)
# %%
def zernike_prop(zern, ampl, plot=True):
    phase = zernike_ansi(zern, D=telescope_diameter)(pupil_grid)
    aberration = np.exp(1j * ampl * phase)
    wf = Wavefront(aberration, wavelength=prop.wl0*1e-6)
    focal_wf = fprop(wf)
    a = normalize(griddata((focal_grid.y, focal_grid.x), focal_wf.electric_field.shaped.ravel(), (mesh.xg / 1e6, mesh.yg/ 1e6), fill_value=0))[PML:-PML, PML:-PML]
    u_out = prop.prop2end(a, remesh_every=0)
    if plot:
        fig, axs = plt.subplots(1, 2)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0].imshow(np.abs(a[np.min(lim[0]):np.max(lim[0]),np.min(lim[1]):np.max(lim[1])]))
        axs[0].set_title(f"Zernike {zern} input -> PSF")
        axs[1].imshow(np.abs(u_out))
        axs[1].set_title("Zernike lantern output")
        plt.show()
    return u_out

# %%
positive_zres = zernike_prop(2, 0.5)
negative_zres = zernike_prop(2, -0.5)
# %%
plt.imshow(np.abs(positive_zres - negative_zres))
# %%
