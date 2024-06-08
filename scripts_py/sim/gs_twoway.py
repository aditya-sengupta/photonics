# %%
from photonics.simulations.lantern_optics import LanternOptics
from photonics.simulations.optics import Optics
from hcipy import imshow_field
from photonics.utils import nanify, zernike_names
from photonics.linearity import plot_linearity
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from tqdm import tqdm

optics = Optics(lantern_fnumber=6.5)
lo = LanternOptics(optics)
lo.nmodes = 18
# %%
def filtered_zernike_to_pl(z, a):
    return lo.forward(optics, filtered_zernike_to_pupil(z, a))
    
def filtered_zernike_to_pupil(z, a):
    pupil = optics.zernike_to_pupil(z, a)
    EM_out = lo.forward(optics, pupil)
    return lo.backward(optics, EM_out, restore_outside=False)

# %%
flat_img = np.abs(lo.plotting_lantern_output(optics.focal_propagator(filtered_zernike_to_pupil(0, 0.0)))) ** 2

def pl_residual(focal_field):
    return np.abs(focal_field) ** 2 - flat_img

def show_pl_residual(focal_field, lim=np.maximum(np.max(flat_img), -np.min(flat_img)), crop=100, ax=None):
    if ax is None:
        ax = plt.gca()
    resid = pl_residual(focal_field)
    r = ax.imshow(resid[crop:-crop,crop:-crop], cmap="seismic", vmin=-lim, vmax=lim)
    ax.axis('off')
    return r

def tracking_GS(z, a, niter, guess=None, restore_outside=False):
    filtered_pupil = filtered_zernike_to_pupil(z, a)
    sim_image = filtered_zernike_to_pl(z, a).intensity
    plotting_sim_image = np.abs(lo.plotting_lantern_output(lo.focal_propagator.forward(filtered_pupil))) ** 2
    result = {
        "actual_injected": optics.zernike_basis.coefficients_for(filtered_pupil.phase),
        "plotting_sim_image": plotting_sim_image,
        "phase_screens": [],
        "focal_fields": [],
        "recon_zernikes": [],
    }
    for i in range(niter):
        if i == 0:
            EM_in, measuredAmplitude_in, measuredAmplitude_out = lo.GS_init(optics, sim_image, guess=guess, restore_outside=restore_outside)
        else:
            EM_in = lo.GS_iteration(optics, EM_in, measuredAmplitude_in, measuredAmplitude_out, restore_outside=restore_outside)
            
        result["phase_screens"].append(EM_in.phase)
        focal_field = lo.plotting_lantern_output(lo.focal_propagator.forward(EM_in))
        result["focal_fields"].append(focal_field)
        result["recon_zernikes"].append(optics.zernike_basis.coefficients_for(EM_in.phase)[:lo.nmodes])
        
    result["recon_zernikes"] = np.array(result["recon_zernikes"])
    return result


def tracking_GS_movie(z, a, niter=20, crop=100, guess=None, restore_outside=False):
    result = tracking_GS(z, a, niter, guess=guess)
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    im0 = axs[0].imshow(nanify(result["phase_screens"][0].shaped, optics.aperture.shaped), vmin=-np.pi/2, vmax=np.pi/2, cmap="RdBu")
    axs[0].axis('off')
    resid = pl_residual(result["focal_fields"][0])
    lim = np.maximum(np.max(resid), -np.min(resid))
    im1 = axs[1].imshow(resid, vmin=-lim, vmax=lim, cmap="seismic")
    axs[1].axis('off')
    
    def animate(t):
        fig.suptitle(f"Gerchberg-Saxton iterations, {zernike_names[z]}, {a} rad, frame {t}")
        recon_mode = result["recon_zernikes"][t][z]
        axs[0].set_title(f"Phase screen, reconstructed {zernike_names[z]} = {recon_mode:.2f} rad")
        im0.set_data(nanify(result["phase_screens"][t].shaped, optics.aperture.shaped))
        ff = result["focal_fields"][t]
        if t > 0:
            mean_change_since_last = np.mean(100 * np.abs(np.abs(ff) ** 2 / np.abs(result["focal_fields"][t-1]) ** 2 - 1))
        else:
            mean_change_since_last = 0.0
        axs[1].set_title(f"PL residual, change since last = {mean_change_since_last:.2f}%")
        im1.set_data(pl_residual(ff)[crop:-crop,crop:-crop])
    
    anim = animation.FuncAnimation(fig, animate, np.arange(len(result["phase_screens"])))
    plt.close(fig)
    return HTML(anim.to_jshtml(default_mode='loop'))

def tracking_GS_zernikes(z, a, niter=20, guess=None, restore_outside=False):
    result = tracking_GS(z, a, niter, guess=guess, restore_outside=restore_outside)
    a_target = result["actual_injected"][z]
    zdecomps = result["recon_zernikes"].T
    plt.hlines(a_target, 0, zdecomps.shape[1] - 1, label="Target")
    for (i, r) in enumerate(zdecomps):
        if i == z:
            plt.plot(r, label="Injected mode", color="k")
        else:
            plt.plot(r, alpha=0.1, color="r")
    plt.xticks(np.arange(0, zdecomps.shape[1], 4))
    plt.xlabel("Gerchberg-Saxton iteration")
    plt.ylabel("Amplitude (rad)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.8)
    plt.suptitle(f"Gerchberg-Saxton iterations, {zernike_names[z]}, {a} rad")

# %%
tracking_GS_movie(3, 0.7, 21)
# %%
tracking_GS_zernikes(3, 0.7)
# %%
