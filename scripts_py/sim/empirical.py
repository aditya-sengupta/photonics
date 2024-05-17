# %%
from IPython import get_ipython
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.optics import Optics
from hcipy import imshow_field, ModeBasis
from photonics.utils import nanify, lmap, norm, corr, rms
import numpy as np
import hcipy as hc
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from itertools import product

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
# %%
optics = Optics(lantern_fnumber=6.5)
lo = LanternOptics(optics)
N = lo.nports
# %%
a = 0.01
zerns_and_amps_for_input = [(0, 0.0)] + [(i, a) for i in range(18)]
input_wavefronts = [optics.focal_propagator(optics.zernike_to_pupil(z, a)) for (z, a) in zerns_and_amps_for_input]
input_basis_fields = [w.electric_field.shaped for w in input_wavefronts]

print(np.linalg.matrix_rank(
    np.array(
        lmap(
            lambda x: np.abs(lo.lantern_coeffs(x)) ** 2,
            input_wavefronts
        )
    )
))

# %%
lantern_basis_fields = ModeBasis(lo.outputs.T)
# %%
A = np.zeros((lo.nports, lo.nports), dtype=np.complex128)
A2obs = np.zeros((lo.nports, lo.nports), dtype=np.complex128)
for i in range(19):
    A[:,i] = lantern_basis_fields.coefficients_for(lo.sanitize_output(input_basis_fields[i]))
    A2obs[:,i] = np.abs(lantern_basis_fields.coefficients_for(lo.sanitize_output(input_basis_fields[i]))) ** 2
# %%
"""
A translates from the zernike-focal basis to the output basis.

What I'm asserting is the following operations are equivalent:

1. Project a PSF onto the zernike-focal basis and multiply by A, to get coefficients on the lantern.
2. Project a PSF onto the lantern basis, to get coefficients on the lantern.

This is the same thing if A is full-rank, which I've confirmed it is. I can't confirm this in reality, but I can confirm if A2obs is full-rank, and these are equivalent. 
"""
# %%
# make (N-1 choose 2) queries

queries = []
for i in trange((N-1)*(N-2)//2):
    z = np.random.randn(N)
    focal_image = optics.focal_propagator(optics.zernike_to_pupil(np.arange(N), z))
    queries.append(focal_image)
# %%
zernike_focal_basis = ModeBasis([lo.sanitize_output(x.electric_field.shaped) for x in input_wavefronts])
# %%
queries_in_lantern_footprint = [lo.sanitize_output(x.electric_field.shaped) for x in queries]
# %%
query_coeffs = [lantern_basis_fields.coefficients_for(x) for x in tqdm(queries_in_lantern_footprint)]
queries_in_zf_basis = [zernike_focal_basis.coefficients_for(x) for x in tqdm(queries_in_lantern_footprint)]
# %%
k = 152
lantern_basis_expression = lantern_basis_fields.linear_combination(query_coeffs[k])
zernike_basis_expression = zernike_focal_basis.linear_combination(queries_in_zf_basis[k])
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
query_cut_to_lantern = lo.input_to_2d(lo.sanitize_output(queries[k].electric_field.shaped))
for ax in axs:
    ax.set_axis_off()
axs[0].imshow(np.abs(query_cut_to_lantern) ** 2, cmap='hot', origin='lower')
axs[0].set_title("PSF cut to lantern input")
axs[1].imshow(np.abs(lo.input_to_2d(lantern_basis_expression)) ** 2, cmap='hot', origin='lower')
axs[1].set_title("PSF in lantern basis")

axs[2].imshow(np.abs(lo.input_to_2d(zernike_basis_expression)) ** 2, cmap='hot', origin='lower')
axs[2].set_title("PSF in Zernike basis")

plt.tight_layout()
plt.show()
# %%
print(f"{float(rms(100 * np.abs(((A @ queries_in_zf_basis[k]) - query_coeffs[k]) / (query_coeffs[k])))):.2f}% error")

# %%
