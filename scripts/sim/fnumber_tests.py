# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import hcipy as hc
from hcipy import imshow_field
from scipy.optimize import minimize
from photonics.lantern_optics import LanternOptics
from itertools import product, repeat
from photonics import PROJECT_ROOT
from tqdm import tqdm
# %%
lo = LanternOptics(f_number=10)
# %%
lo.setup_hcipy(f_number=6.5)
# %%
<<<<<<< HEAD:scripts/sim/fnumber_tests.py
lo.make_command_matrix(nzern=9)
=======
lo.make_intcmd(nzern=9)
# %%
>>>>>>> b9334baeaae2ffb972e51444f981bf0284e999b4:scripts/sim/backwards_prop_19.py
amplitudes, linearity_responses = lo.make_linearity(nzern=9, lim=1.0, step=0.05)
# %%
lo.show_linearity(amplitudes, linearity_responses)
# %%
def linearity_loss(amplitudes, linearity_responses):
    comparison = np.eye(linearity_responses.shape[2])[:, np.newaxis, :]
    return np.sum((comparison * amplitudes[np.newaxis, :, np.newaxis] - linearity_responses) ** 2)

# %%
linearity_loss(amplitudes, linearity_responses)
# %%
def fnumber_objective(f, nzern=9):
    print(f)
    lo.setup_hcipy(f_number=f)
    lo.make_intcmd(nzern=nzern)
    loss = linearity_loss(*lo.make_linearity(nzern=nzern, lim=0.5, step=0.1))
    print(loss)
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
    lo.setup_hcipy(f_number=f)
    psf_on_input = np.array(lo.focal_wf_ref.electric_field.shaped[lo.input_footprint])
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
plt.savefig(PROJECT_ROOT + "/figures/coupling_efficiency.png", dpi=600)

# %%
f_shortlist = np.arange(3, 5.1, 0.1)
# %%
linearity_arrays = []
linearity_loss_vals = []
# %%
for f in f_shortlist:
    print(f)
    lo.setup_hcipy(f_number=f)
    lo.make_intcmd(nzern=18)
    amplitudes, linearity_arr = lo.make_linearity(nzern=18, lim=0.5, step=0.1)
    linearity_arrays.append(linearity_arr)
    linearity_loss_vals.append(linearity_loss(amplitudes, linearity_arr))
    
# %%
plt.semilogy(f_shortlist, linearity_loss_vals)
plt.xlabel("f-number")
plt.ylabel("Linearity loss value")
plt.title(f"The optimal f-number is {f_shortlist[np.argmin(linearity_loss_vals)]:.1f}")
plt.savefig(PROJECT_ROOT + "/figures/optimal_f.png", dpi=600)
# %%
nzern = 18 # linearity_arrays[0].shape[0]
fig, axs = plt.subplots(int(np.ceil(nzern // 3)), 3, sharex=True, sharey=True, figsize=(9, 18))
plt.suptitle("Photonic lantern linearity curves (rad)")
linearity_arrays_cr = np.array(linearity_arrays)[np.argsort(f_shortlist)]
f_to_test_cr = np.sort(f_shortlist)
def rescaled(x, s):
    return ((x - np.min(x)) / (np.max(x) - np.min(x))) * s + (1 - s) / 2

for i in range(nzern):
    r, c = i // 3, i % 3
    axs[r][c].set_prop_cycle(plt.cycler('color', plt.cm.magma(rescaled(f_to_test_cr, 0.6))))
    axs[r][c].set_ylim([min(amplitudes), max(amplitudes)])
    axs[r][c].title.set_text(f"Z{i+1}")
    axs[r][c].plot(amplitudes, amplitudes, '--k')
    for (j,f) in enumerate(f_to_test_cr):
        axs[r][c].plot(amplitudes, linearity_arrays_cr[j][i,:,i], label=(f"f={f:.1f}" if j % 1 == 0 else None))
plt.legend(bbox_to_anchor=(1.04, 0.8), loc="lower left")
plt.savefig(PROJECT_ROOT + f"/figures/linearity_fsweep_z{nzern}_3_5.png", dpi=600)
plt.show()

# %%
def psf_entrance_scanning(f_number):
    lo.setup_hcipy(f_number=f_number)
    norm_val = np.linalg.norm(lo.focal_wf_ref.intensity)
    zvals = np.arange(1, 10)
    amp_vals = np.arange(-1, 1.0001, 0.05)
    input_psfs = [lo.zernike_to_focal(*k) for k in product(zvals, amp_vals)]
    zvals_repeat = [list(repeat(z, len(amp_vals))) for z in zvals]
    amp_vals_repeat = list(repeat(amp_vals, len(zvals)))
    zvals = [x for xs in zvals_repeat for x in xs]
    amp_vals = [x for xs in amp_vals_repeat for x in xs]
    scan = [
        np.abs(lo.lantern_output(input_psf)[1]) ** 2
        for input_psf in input_psfs
    ]
    coeffs = [
        np.abs(lo.lantern_output(input_psf)[0]) ** 2
        for input_psf in input_psfs
    ]
    max_coeff = np.max(np.max(coeffs))
    coeffs = [x / max_coeff for x in coeffs]
    lantern_image = np.abs(sum(c * lf for (c, lf) in zip(coeffs[0], lo.plotting_launch_fields))) ** (1/2)
    fig = plt.figure(figsize=(10, 3))
    fig.subplots_adjust(top=0.8)
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(np.log10(input_psfs[0].intensity.shaped[lo.extent_x[0]:lo.extent_x[1],lo.extent_y[0]:lo.extent_y[1]] / norm_val), vmin=-2)
    plt.title("Input PSF")
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(scan[0])
    plt.title("Projection onto lantern basis")
    plt.subplot(1, 3, 3)
    # im3 = plt.bar(np.arange(19), coeffs[0])
    # plt.ylim(0, 1)
    im3 = plt.imshow(lantern_image)
    plt.title("Projection coeffs/lantern outputs")
    def animate(t):
        fig.suptitle(f"Zernike {zvals[t]}, amplitude {amp_vals[t]:.2f}, f/{lo.f_number} lantern projection", y=1)
        im1.set_data(np.log10(input_psfs[t].intensity.shaped[lo.extent_x[0]:lo.extent_x[1],lo.extent_y[0]:lo.extent_y[1]] / norm_val))
        im2.set_data(scan[t])
        im3.set_data(np.abs(sum(c * lf for (c, lf) in zip(coeffs[t], lo.plotting_launch_fields))) ** (1/2))
        return [im1, im2, im3]
    anim = animation.FuncAnimation(fig, animate, np.arange(len(scan)))
    plt.close(fig)
    HTML(anim.to_jshtml(default_mode='loop'))
    anim.save(PROJECT_ROOT + f"/figures/psf_entrance_scanning_f{lo.f_number}.mp4")
    
# %%
psf_entrance_scanning(6.5)
# %%
for fv in [4, 6.5, 9]:
    idx = np.argmin(np.abs(fv - f_shortlist))
    lo.setup_hcipy(f_number=fv)
    linearity_array = linearity_arrays[idx]
    lo.show_linearity(amplitudes, linearity_array)
# %%
