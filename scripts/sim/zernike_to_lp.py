# %%
from hcipy import *
from hcipy.mode_basis import zernike_ansi

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tqdm import trange

backslash = lambda a, b: np.linalg.lstsq(a, b, rcond=-1)[0]

# %%
wavelength = 635.0e-9
telescope_diameter = 1.0
pupil_grid = make_pupil_grid(60, telescope_diameter)
aperture = evaluate_supersampled(make_circular_aperture(telescope_diameter), pupil_grid, 6)

coupling_fraction = 0.3 # unaberrated PSF radius / input core radius
core_radius = 1.5e-6 # m
cladding_radius = 8.0e-6 # m; matching Lightbeam, but for now just step-index
target_psf_radius = coupling_fraction * core_radius

spatial_resolution = wavelength / telescope_diameter # m/m = fraction
focal_grid = make_focal_grid(q=10, num_airy=16, spatial_resolution=spatial_resolution)
prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=2 * target_psf_radius / spatial_resolution)
norm = np.max(prop(Wavefront(aperture, wavelength=wavelength)).intensity)
# %%
lp_basis = make_lp_modes(focal_grid, 8.5, core_radius)
A = lp_basis.transformation_matrix
# A.dtype = np.complex64
A = np.nan_to_num(A, 0.0)
Q, R = np.linalg.qr(A)
# %%
ref_focal_wf = prop(Wavefront(aperture, wavelength=wavelength))
# %%
def plot_coupling_for_zernike(zernike, amplitude):
    assert isinstance(zernike, int)
    assert isinstance(amplitude, float)
    phase = zernike_ansi(zernike, D=telescope_diameter)(pupil_grid)
    aberration = np.exp(1j * amplitude * phase)
    wavefront = Wavefront(aperture * aberration, wavelength=wavelength)
    focal_wf = prop(wavefront)
    projection = Field(Q @ Q.T.dot(focal_wf.electric_field), focal_grid)
    residual = focal_wf.electric_field - projection
    fig = plt.figure(figsize=(9.75, 3))

    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1,3),
                    axes_pad=0.15,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )

    for (ax, data, title) in zip(
        grid,
        [
            np.log10(focal_wf.intensity / norm),
            np.log10(np.abs(projection ** 2) / norm),
            np.log10(np.abs(residual ** 2) / norm)
        ],
        [
            "Input PSF",
            "Coupled light",
            "Residual light"
        ]
    ):
        im = imshow_field(data, ax=ax,vmin=-5, vmax=0)
        ax.set_title(title)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    plt.show()
    print(np.sum(np.abs(projection ** 2)) / np.sum(focal_wf.intensity))

# %%
plot_coupling_for_zernike(2, 0.5)

# %%
