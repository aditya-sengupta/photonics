# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
import os
if "CONDA_PREFIX" in os.environ:
    os.environ.__delitem__("CONDA_PREFIX")

import numpy as np
import hcipy as hc
from copy import copy
from matplotlib import pyplot as plt
from hcipy import imshow_field
from photonics import DATA_PATH
from photonics.utils import rms, nanify, lmap
from photonics.second_stage import correction
from juliacall import Main as jl
jl.seval("using Flux")
jl.seval("using JLD2")
# %%
sso = SecondStageOptics()
max_amp_nn = 0.5
model_fname = f"pl_nn_{max_amp_nn}"
model_state = jl.JLD2.load(DATA_PATH + f"/pl_nn/{model_fname}.jld2", "model_state")
if model_fname.endswith("19"):
    nzern = 19
    model = jl.Chain(
        jl.Dense(19, 2000, jl.relu),
        jl.Dense(2000, 100, jl.relu),
        jl.Dropout(0.2),
        jl.Dense(100, 19)
    )
else:
    nzern = 9
    model = jl.Chain(
        jl.Dense(19, 2000, jl.relu),
        jl.Dense(2000, 100, jl.relu),
        jl.Dense(100, 9)
    )

jl.Flux.loadmodel_b(model, model_state)
ymin, ymax = (lambda x: (np.min(x), np.max(x)))(np.abs(np.load(DATA_PATH + "/sim_trainsets/sim_trainset_lanterns_240502_1947.npy")) ** 2)
zernike_basis = hc.mode_basis.make_zernike_basis(nzern, sso.telescope_diameter, sso.pupil_grid)

def nn_inject_recover(zernikes, amplitudes):
    phase_screen = sso.zernike_to_phase(zernikes, amplitudes)
    psf = sso.focal_propagator(
        hc.Wavefront(sso.aperture * np.exp(1j * phase_screen), sso.wl)
    )
    post_lantern_coeffs = sso.lantern_optics.lantern_coeffs(psf)
    intensities = np.abs(post_lantern_coeffs) ** 2
    norm_intensities = ((intensities - ymin) / (ymax - ymin)).astype(np.float32)
    reconstructed_zernike_coeffs = np.array(model(norm_intensities)) * 0.5 - 0.25
    return reconstructed_zernike_coeffs

def reconstruct(phase_screen, plot=True):
    """
    Takes in a post-DM phase screen and injects and recovers it from the lantern.
    """
    phase_screen_coeffs = zernike_basis.coefficients_for(phase_screen - np.mean(phase_screen))
    phase_screen_projected = zernike_basis.linear_combination(phase_screen_coeffs)
    psf = sso.focal_propagator(
        hc.Wavefront(sso.aperture * np.exp(1j * phase_screen_projected), sso.wl)
    )
    post_lantern_coeffs = sso.lantern_optics.lantern_coeffs(psf)
    intensities = np.abs(post_lantern_coeffs) ** 2
    norm_intensities = ((intensities - ymin) / (ymax - ymin)).astype(np.float32)
    reconstructed_zernike_coeffs = model(norm_intensities) 
    # sso.lantern_optics.command_matrix @ (np.abs(post_lantern_coeffs) ** 2 - sso.lantern_optics.flat_amp)
    reconstructed_phase = zernike_basis.linear_combination(max_amp_nn * np.array(reconstructed_zernike_coeffs) - max_amp_nn/2)
    
    if plot:
        _, axs = plt.subplots(1, 3)
        for ax in axs:
            ax.axis('off')
        projected_zeroed = nanify(phase_screen_projected, sso.aperture)
        reconstructed_zeroed = nanify(reconstructed_phase, sso.aperture)
        vmin = np.minimum(np.nanmin(projected_zeroed), np.nanmin(reconstructed_zeroed))
        vmax = np.maximum(np.nanmax(projected_zeroed), np.nanmax(reconstructed_zeroed))
        imshow_field(np.log10(psf.intensity / sso.norm), ax=axs[0], vmin=-5)
        axs[0].set_title("PSF")
        imshow_field(projected_zeroed, ax=axs[1], vmin=vmin, vmax=vmax)
        axs[1].set_title(f"PL input phase, {rms(phase_screen_coeffs):.2f} rad", fontsize=10)
        imshow_field(reconstructed_zeroed, ax=axs[2], vmin=vmin, vmax=vmax)
        axs[2].set_title("PL recon. phase", fontsize=10)
        plt.show()
        
# %%
r0 = 0.1
sso.turbulence_setup(fried_parameter=r0)
# %%
sso.layer.reset()
sso.deformable_mirror.flatten()
correction_results = sso.pyramid_correction(num_iterations=10)
correction_results["strehl_ratios"]
# %%
second_stage_phase_screens = [x.phase * sso.aperture for x in correction_results["wavefronts_after_dm"]]
print(f"Second-stage WF reconstruction, r0 = {r0} m")
for i in range(0, 200, 20):
    reconstruct(second_stage_phase_screens[i])

# %%
