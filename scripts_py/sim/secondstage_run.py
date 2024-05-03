# %%
# ask about adding this to default profile on threadripper
%load_ext autoreload
%autoreload 2

# %%
import os
os.environ.__delitem__("CONDA_PREFIX")

def lmap(f, x):
    return list(map(f, x))

# %%
import numpy as np
import hcipy as hc
from matplotlib import pyplot as plt
from hcipy import imshow_field
from photonics import DATA_PATH
from photonics.second_stage_optics import SecondStageOptics
from juliacall import Main as jl
jl.seval("using Flux")
jl.seval("using JLD2")
# %%
sso = SecondStageOptics()
correction_results = sso.pyramid_correction(gain=0.1)
dm_phase_screens = [x * 2 * np.pi / (sso.wl) for x in correction_results["dm_shapes"]]
second_stage_phase_screens = [p - d for (p, d) in zip(correction_results["phases_for"], dm_phase_screens)]
lantern_inputs = correction_results["point_spread_functions"] 
second_stage_aperture_phase_screens = [sso.aperture * s for s in second_stage_phase_screens]
zernike_basis = hc.mode_basis.make_zernike_basis(9, sso.telescope_diameter, sso.pupil_grid)
# %%
model_state = jl.JLD2.load(DATA_PATH + "/pl_nn/pl_nn_0.25.jld2", "model_state")
model = jl.Chain(
    jl.Dense(19, 2000, jl.relu),
    jl.Dense(2000, 100, jl.relu),
    jl.Dense(100, 9)
)
jl.Flux.loadmodel_b(model, model_state)
ymin, ymax = (lambda x: (np.min(x), np.max(x)))(np.abs(np.load(DATA_PATH + "/sim_trainsets/sim_trainset_lanterns_240428_1706.npy")) ** 2)
# %%
def reconstruct(phase_screen, plot=True):
    """
    Takes in a post-DM phase screen and injects and recovers it from the lantern.
    """
    psf = sso.focal_propagator(
        hc.Wavefront(sso.aperture * np.exp(1j * phase_screen), sso.wl)
    )
    post_lantern_coeffs = sso.lantern_optics.lantern_coeffs(psf)
    intensities = np.abs(post_lantern_coeffs) ** 2
    norm_intensities = ((intensities - ymin) / (ymax - ymin)).astype(np.float32)
    reconstructed_zernike_coeffs = model(norm_intensities) 
    # sso.lantern_optics.command_matrix @ (np.abs(post_lantern_coeffs) ** 2 - sso.lantern_optics.flat_amp)
    reconstructed_phase = zernike_basis.linear_combination(np.array(reconstructed_zernike_coeffs) - 0.5)
    
    if plot:
        fig, axs = plt.subplots(1, 3)
        imshow_field(np.log10(psf.intensity), ax=axs[0])
        imshow_field(phase_screen, ax=axs[1])
        imshow_field(reconstructed_phase, ax=axs[2], vmin=np.min(phase_screen), vmax=np.max(phase_screen))
        plt.show()
        
# %%

# %%
reconstruct(sso.lantern_optics.zernike_to_phase(np.arange(1, 10), second_stage_phase_coeffs[159]))
# %%
second_stage_phase_coeffs = np.array(lmap(zernike_basis.coefficients_for, second_stage_aperture_phase_screens))
# %%
lantern_outputs = np.abs(np.array(lmap(sso.lantern_optics.lantern_coeffs, lantern_inputs)).T) ** 2
# %%
sso.lantern_optics.make_command_matrix()
# %%
lantern_reconstructed_zernikes = (sso.lantern_optics.command_matrix @ lantern_outputs)
# %%
lantern_reconstructed_phase_screens = lmap(zernike_basis.linear_combination, lantern_reconstructed_zernikes.T)
# %%
idx = 141
fig, axs = plt.subplots(1, 2, figsize=(10,5))
imshow_field(second_stage_aperture_phase_screens[idx], ax=axs[0])
imshow_field(lantern_reconstructed_phase_screens[idx], ax=axs[1])
# %%
