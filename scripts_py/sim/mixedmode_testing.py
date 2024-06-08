# %%
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.pyramid_optics import PyramidOptics
from photonics.simulations.optics import Optics
from hcipy import imshow_field
from photonics.utils import nanify, zernike_names, rms, lmap
from photonics.linearity import plot_linearity
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from tqdm import tqdm
from photonics.utils import DATA_PATH
from os import path

# %%
optics = Optics(lantern_fnumber=6.5)
pyr = PyramidOptics(optics)
lo = LanternOptics(optics)
lo.nmodes = 9
# %%
test_phase_screens = np.random.uniform(-1, 1, (1000,9))
test_normalizations = np.random.uniform(0, 1, (1000,))
current_normalizations = np.sqrt(np.sum(test_phase_screens ** 2, axis=1))
test_phase_screens /= current_normalizations[:,np.newaxis]
test_phase_screens *= test_normalizations[:,np.newaxis]
rms_inputs = np.sqrt(np.sum(test_phase_screens ** 2, axis=1))
plt.hist(rms_inputs)
# %%
pupils = [optics.zernike_to_pupil(np.arange(9), x) for x in test_phase_screens]
focals = lmap(optics.focal_propagator, pupils)
coeffs = np.array(lmap(lo.lantern_coeffs, focals))
# %%
readings = np.abs(coeffs) ** 2 - lo.image_ref[np.newaxis,:]

# %%
linear_reconstructions = readings @ lo.command_matrix.T / (optics.wl / (4 * np.pi))
# %%
gs_reconstructions = []
for p in tqdm(pupils):
    gs_reconstructions.append(lo.GS(optics, lo.forward(optics, p).intensity))
# %%
residuals = linear_reconstructions - test_phase_screens
rms_residuals = np.sqrt(np.sum(residuals ** 2, axis=1))

# %%
gs_zernikes = lmap(lambda x: optics.zernike_basis.coefficients_for(x.phase), gs_reconstructions)

# %%
gs_residuals = np.array(gs_zernikes)[:,:lo.nmodes] - test_phase_screens
gs_rms_residuals = np.sqrt(np.sum(gs_residuals ** 2, axis=1))
# %%
pyramid_linear_reconstructions = lmap(pyr.reconstruct, pupils)

# %%
pyramid_linear_reconstructions_ = np.array(pyramid_linear_reconstructions) / (optics.wl / (4 * np.pi))
# %%
pyramid_residuals = pyramid_linear_reconstructions_[:,:lo.nmodes] - test_phase_screens
pyramid_rms_residuals = np.sqrt(np.sum(pyramid_residuals ** 2, axis=1))
# %%
np.save(path.join(DATA_PATH, "nonlinear_test", "zernike_inputs.npy"), test_phase_screens)
np.save(path.join(DATA_PATH, "nonlinear_test", "output_coeffs.npy"), coeffs)
np.save(path.join(DATA_PATH, "nonlinear_test", "pyramid_rms_residuals.npy"), pyramid_rms_residuals)
np.save(path.join(DATA_PATH, "nonlinear_test", "rms_inputs.npy"), rms_inputs)
np.save(path.join(DATA_PATH, "nonlinear_test", "rms_residuals.npy"), rms_residuals)
np.save(path.join(DATA_PATH, "nonlinear_test", "gs_rms_residuals.npy"), gs_rms_residuals)
# %%
plt.scatter(rms_inputs, rms_residuals, s=3, label="Linear reconstructor")
plt.scatter(rms_inputs, gs_rms_residuals, s=3, label="Gerchberg-Saxton")
plt.xlabel("Injected aberration RMS (rad)")
plt.ylabel("Reconstruction residual RMS (rad)")
plt.legend()
# %%
