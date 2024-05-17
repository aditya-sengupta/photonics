# %%
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import hcipy as hc
from hcipy import imshow_field
from photonics import Optics, LanternOptics, make_command_matrix, imshow_psf
from itertools import product, repeat
from photonics import PROJECT_ROOT, zernike_names, nanify
from tqdm import tqdm
# %%
optics = Optics(lantern_fnumber=6.5)
lo = LanternOptics(optics)
nzern = lo.nmodes
# %%
make_command_matrix(optics.deformable_mirror, lo, optics.wf)
amplitudes, linearity_responses = lo.make_linearity(optics, lim=1.0, step=0.05)
# %%
lo.show_linearity(amplitudes, linearity_responses)
# %%
def linearity_loss(amplitudes, linearity_responses):
    comparison = np.eye(linearity_responses.shape[2])[:, np.newaxis, :]
    return np.sum((comparison * amplitudes[np.newaxis, :, np.newaxis] - linearity_responses) ** 2)

# %%
linearity_loss(amplitudes, linearity_responses)

# %%
def fnumber_objective(f):
    optics = Optics(lantern_fnumber=f)
    lo.focal_propagator = optics.focal_propagator
    make_command_matrix(optics.deformable_mirror, lo, optics.wf)
    loss = linearity_loss(*lo.make_linearity(optics, lim=0.5, step=0.1))
    return loss
# %%
def norm(a, b):
    return np.sum(a * np.conj(b))
def corr(a, b):
    return np.real(np.abs(norm(a, b))) / np.sqrt(norm(a, a) * norm(b, b))
# %%
f_to_test = np.arange(0.1, 20.1, 0.1)
corrs = []
for f in tqdm(f_to_test):
    optics = Optics(lantern_fnumber=f)
    lo.focal_propagator = optics.focal_propagator
    psf_on_input = np.array(optics.im_ref.electric_field.shaped[lo.input_footprint])
    projected = lo.outputs.T @ (lo.projector @ psf_on_input)
    corrs.append(corr(psf_on_input, projected))

# %%
corrs = np.real(corrs)
# %%
plt.semilogy(f_to_test, 1 - np.array(corrs))
plt.gca().invert_yaxis()
plt.yticks([1e0, 1e-1, 1e-2, 1e-3], [1-1e0, 1-1e-1, 1-1e-2, 1-1e-3])
plt.xlabel("f-number")
plt.ylabel("Coupling efficiency")
plt.savefig(join(PROJECT_ROOT, "figures", "coupling_efficiency.png", dpi=600))

# %%
f_shortlist = np.arange(3.0, 10.0, 0.1)
# %%
linearity_arrays = {}
linearity_loss_vals = {}
# %%
for f in tqdm(f_shortlist):
    optics = Optics(lantern_fnumber=f)
    lo.focal_propagator = optics.focal_propagator
    make_command_matrix(optics.deformable_mirror, lo, optics.wf)
    amplitudes, linearity_arr = lo.make_linearity(optics, lim=0.5, step=0.1)
    loss = linearity_loss(amplitudes, linearity_arr)
    linearity_arrays[f] = linearity_arr
    linearity_loss_vals[f] = linearity_loss(amplitudes, linearity_arr)
    
# %%
linearity_losses_list = [v for (_, v) in sorted(linearity_loss_vals.items())]
optimal_f = min(linearity_loss_vals, key=linearity_loss_vals.get)
plt.semilogy(f_shortlist, linearity_losses_list)
plt.xlabel("f-number")
plt.ylabel("Linearity loss value")
plt.title(f"The optimal f-number is {optimal_f:.1f}")
plt.savefig(join(PROJECT_ROOT, "figures", "optimal_f.png"), dpi=600)
# %%
fig, axs = plt.subplots(int(np.ceil(nzern // 3)), 3, sharex=True, sharey=True, figsize=(9, 9))
plt.suptitle("Photonic lantern linearity curves (rad)")
linearity_arrays_cr = [v for (_, v) in sorted(linearity_arrays.items())]
f_to_test_cr = sorted(linearity_arrays.keys())
def rescaled(x, s):
    return ((x - np.min(x)) / (np.max(x) - np.min(x))) * s + (1 - s) / 2

for i in range(nzern):
    r, c = i // 3, i % 3
    axs[r][c].set_prop_cycle(plt.cycler('color', plt.cm.magma(rescaled(f_to_test_cr, 0.6))))
    axs[r][c].set_ylim([min(amplitudes), max(amplitudes)])
    axs[r][c].title.set_text(zernike_names[i])
    axs[r][c].plot(amplitudes, amplitudes, '--k')
    for (j,f) in enumerate(f_to_test_cr):
        axs[r][c].plot(amplitudes, linearity_arrays_cr[j][i,:,i], label=(f"f={f:.1f}" if j % 10 == 0 else None), alpha=0.4)
plt.legend(bbox_to_anchor=(1.04, 0.8), loc="lower left")
plt.savefig(join(PROJECT_ROOT, "figures", f"linearity_fsweep_z{nzern}_3_5.png"), dpi=600)
plt.show()
# %%
def psf_entrance_scanning(f_number):
    optics = Optics(lantern_fnumber=f_number)
    lo.focal_propagator = optics.focal_propagator
    norm_val = np.linalg.norm(optics.im_ref.intensity)
    zvals = np.arange(10)
    amp_vals = np.arange(-1, 1.0001, 0.1)
    input_psfs = [optics.focal_propagator(optics.zernike_to_pupil(*k)) for k in product(zvals, amp_vals)]
    zvals_repeat = [list(repeat(z, len(amp_vals))) for z in zvals]
    amp_vals_repeat = list(repeat(amp_vals, len(zvals)))
    zvals = [x for xs in zvals_repeat for x in xs]
    amp_vals = [x for xs in amp_vals_repeat for x in xs]
    coeffs = [
        np.abs(lo.lantern_output(input_psf)[0]) ** 2
        for input_psf in input_psfs
    ]
    scan = [
        np.abs(lo.input_to_2d(lo.lantern_output(input_psf)[0] @ lo.outputs)) ** 2
        for input_psf in input_psfs
    ]
    max_coeff = np.max(np.max(coeffs))
    coeffs = [x / max_coeff for x in coeffs]
    lantern_image = np.abs(sum(c * lf for (c, lf) in zip(coeffs[0], lo.plotting_launch_fields))) ** (1/2)
    fig = plt.figure(figsize=(9, 3))
    fig.subplots_adjust(top=0.8)
    plt.subplot(1, 3, 1)
    ax = plt.gca()
    ax.set_axis_off()
    im1 = plt.imshow(np.log10(input_psfs[0].intensity.shaped[lo.extent_x[0]:lo.extent_x[1],lo.extent_y[0]:lo.extent_y[1]] / norm_val), vmin=-2)
    plt.title("Input PSF")
    plt.subplot(1, 3, 2)
    ax = plt.gca()
    ax.set_axis_off()
    im2 = plt.imshow(scan[0])
    plt.title("What the lantern sees")
    plt.subplot(1, 3, 3)
    ax = plt.gca()
    ax.set_axis_off()
    im3 = plt.imshow(lantern_image)
    plt.title("Lantern output")
    def animate(t):
        fig.suptitle(f"{zernike_names[zvals[t]].title()}, amplitude {amp_vals[t]:.2f} rad, f/{lo.f_number} lantern projection", y=1)
        im1.set_data(np.log10(input_psfs[t].intensity.shaped[lo.extent_x[0]:lo.extent_x[1],lo.extent_y[0]:lo.extent_y[1]] / norm_val))
        im2.set_data(scan[t])
        im3.set_data(np.abs(sum(c * lf for (c, lf) in zip(coeffs[t], lo.plotting_launch_fields))) ** (1/2))
        return [im1, im2, im3]
    anim = animation.FuncAnimation(fig, animate, np.arange(len(scan)))
    plt.close(fig)
    HTML(anim.to_jshtml(default_mode='loop'))
    anim.save(PROJECT_ROOT + f"/figures/psf_entrance_scanning_f{lo.f_number}.mp4", dpi=600)
    
# %%
psf_entrance_scanning(6.5)
# %%
for fv in [4, 6.5, 9]:
    idx = np.argmin(np.abs(fv - f_shortlist))
    lo.setup_hcipy(f_number=fv)
    linearity_array = linearity_arrays[idx]
    lo.show_linearity(amplitudes, linearity_array)

# %%
plt.rc('font', family='serif',size=12)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif" : "cmr10",
    "axes.formatter.use_mathtext" : True
})
fnumbers = np.arange(3.5, 12.1, 0.1)
injected_idx = 4
injected_amp = 0.4
optics = Optics(lantern_fnumber=4.0)
lantern_mask = np.array(np.ones_like(optics.im_ref.intensity.shaped)) * 1e-10
lantern_mask[lo.input_footprint] = 1.0
lo.focal_propagator = optics.focal_propagator
optics.deformable_mirror.flatten()
optics.deformable_mirror.actuators[injected_idx] = injected_amp * optics.wl / (4 * np.pi)
injected_pupil = optics.deformable_mirror.forward(optics.wf)
injected_psf = optics.focal_propagator(injected_pupil)
recovered_zernikes = lo.command_matrix.dot(lo.readout(injected_pupil) - lo.image_ref)
optics.deformable_mirror.actuators[:nzern] = recovered_zernikes
recovered_pupil = optics.deformable_mirror.forward(optics.wf)
recovered_psf = optics.focal_propagator(recovered_pupil)
fig, axs = plt.subplots(2, 2, figsize=(5,5), dpi=400)
for axr in axs:
    for ax in axr:
        ax.axis('off')
        # ax.set_xticks([])
        # ax.set_yticks([])
im0 = axs[0][0].imshow(
    np.log10(injected_psf.intensity.shaped / np.max(injected_psf.intensity))[219:293, 219:293],
    vmin=-4,
)
axs[0][0].set_title("Injected PSF")
injected_to_plot = np.log10((injected_psf.intensity.shaped * lantern_mask) / np.max(injected_psf.intensity))[219:293, 219:293]
im1 = axs[0][1].imshow(
    injected_to_plot,
    vmin=-4
)
axs[0][1].set_title("What the lantern sees")
recovered = recovered_zernikes[injected_idx] / (optics.wl / (4 * np.pi))
crosstalk = recovered_zernikes[2] / (optics.wl / (4 * np.pi))
axs[1][0].set_title("Injected phase")
im2 = axs[1][0].imshow(
    injected_pupil.phase.shaped,
    vmin=-np.pi/4, vmax=np.pi/4,
    cmap="RdBu"
)
axs[1][1].set_title("Lantern measured phase")
im3 = axs[1][1].imshow(
    recovered_pupil.phase.shaped,
    vmin=-np.pi/4, vmax=np.pi/4,
    cmap="RdBu"
)
def animate(t):
    fig.suptitle(f"Photonic lantern reconstruction at f/{fnumbers[t]:.1f}")
    optics = Optics(lantern_fnumber=fnumbers[t])
    lo.focal_propagator = optics.focal_propagator
    make_command_matrix(optics.deformable_mirror, lo, optics.wf)
    optics.deformable_mirror.flatten()
    optics.deformable_mirror.actuators[injected_idx] = injected_amp * optics.wl / (4 * np.pi)
    injected_pupil = optics.deformable_mirror.forward(optics.wf)
    injected_psf = optics.focal_propagator(injected_pupil)
    recovered_zernikes = lo.command_matrix.dot(lo.readout(injected_pupil) - lo.image_ref)
    optics.deformable_mirror.actuators[:nzern] = recovered_zernikes
    recovered_pupil = optics.deformable_mirror.forward(optics.wf)
    r = np.array(injected_psf.intensity.shaped)
    central_idx = np.unravel_index(np.argmax(np.array(r)), r.shape)
    imin_x, imax_x = central_idx[1] - 37, central_idx[1] + 37
    imin_y, imax_y = central_idx[0] - 37, central_idx[0] + 37
    cropped_grid = np.indices((imax_y - imin_y, imax_x - imin_x))
    cropped_cen_y, cropped_cen_x = (imax_y - imin_y) // 2, (imax_x - imin_x) // 2
    lantern_aperture = np.zeros((imax_y - imin_y, imax_x - imin_x))
    lantern_aperture[(cropped_grid[0] - cropped_cen_y) ** 2 + (cropped_grid[1] - cropped_cen_x) ** 2 <= (cropped_cen_y - 20) ** 2] = 1.0
    im0.set_data(np.log10(injected_psf.intensity.shaped / np.max(injected_psf.intensity))[imin_y:imax_y, imin_x:imax_x])
    injected_to_plot = np.log10((injected_psf.intensity.shaped) / np.max(injected_psf.intensity))[imin_y:imax_y, imin_y:imax_y]
    injected_to_plot[np.where(lantern_aperture == 0.0)] = np.min(injected_to_plot)
    im1.set_data(injected_to_plot)
    im2.set_data(nanify(injected_pupil.phase, optics.aperture).shaped)
    im3.set_data(nanify(recovered_pupil.phase, optics.aperture).shaped)
    """recovered = recovered_zernikes[injected_idx] / (optics.wl / (4 * np.pi))
    crosstalk = recovered_zernikes[2] / (optics.wl / (4 * np.pi))
    axs[2].clear()
    axs[2].scatter(recovered, crosstalk)
    axs[2].scatter([0.5], [0], c='k')
    axs[2].annotate("Target", [0.5, 0.01])
    axs[2].annotate("Recovered", [recovered + 0.01, crosstalk + 0.01])
    axs[2].set_xlabel("Injected mode (rad)")
    axs[2].set_ylabel("Crosstalk (rad)")
    axs[2].set_xlim([injected_amp - 0.3, injected_amp + 0.3])
    axs[2].set_ylim([-0.25, 0.25])"""
# %%
anim = animation.FuncAnimation(fig, animate, np.arange(len(fnumbers)))
# %%
HTML(anim.to_jshtml(default_mode='loop'))

# %%
anim.save(join(PROJECT_ROOT, "figures", "fnumber_zooming.mp4"), dpi=600)
# %%
