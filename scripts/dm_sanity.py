# %%
import numpy as np
import matplotlib.pyplot as plt
import hcipy

from hcipy import *
from hcipy.mode_basis import zernike_ansi

def intensity_plot(f: hcipy.Field, **kwargs):
    return imshow_field(np.log10(f / f.max()), vmin=-5, **kwargs)

pupil_grid = make_pupil_grid(256)
aperture = make_circular_aperture(1)(pupil_grid)
phase = zernike_ansi(zern)(pupil_grid)
aberration = np.exp(1j * amp * phase) * aperture
focal_grid = make_focal_grid(q=64, num_airy=16)
prop = FraunhoferPropagator(pupil_grid, focal_grid)
bprop = FraunhoferPropagator(focal_grid, pupil_grid)

def zernike_phase(zern):
    return zernike_ansi(zern)(pupil_grid)

def zernike_ampl(zern, amp):
    phase = zernike_phase(zern)
    return np.exp(1j * amp * phase) * aperture

def focal_wavefront(zern, amp):
    return prop.forward(Wavefront(zernike_ampl(zern, amp)))

def wavefront_sum_intensity(wf1, wf2):
    return np.abs(
        wf1.electric_field + wf2.electric_field
    ) ** 2

def unwrap_phase(f: hcipy.Field):
    v = f.shaped
    for i in range(len(f.shape)):
        v = np.unwrap(v, axis=i)
    return Field(np.mod(v.ravel(), 2*np.pi), f.grid)

intensity_sum_plot = lambda wf1, wf2: intensity_plot(wavefront_sum_intensity(wf1, wf2))
# if you did this naively you'd just have

# %%
amp_large = 5.0
intensity_sum_plot(
    focal_image(3, amp_large), 
    focal_image(6, amp_large)
)
plt.title(f"Tip + tilt with amplitude = {amp_large}")

# %%
# and this looks like two PSFs, so not something we can easily produce with a DM. But!

amp_small = 1.2
intensity_sum_plot(
    focal_image(3, amp_small), 
    focal_image(6, amp_small)
)
plt.title(f"Tip + tilt with amplitude = {amp_small}")

# turn down the amplitude, and you get something that looks more like *one* PSF with a higher-order distortion. Let's try back-propagation of sums to Zernike space. First, with the large amplitude:

# %%
truth = Wavefront(focal_image(2, amp_large).electric_field + focal_image(3, amp_large).electric_field)
recon = prop.backward(truth)
imshow_field(recon.phase * aperture, mask=aperture)
plt.title(f"Zernike mode sums with amplitude = {amp_large}")
plt.colorbar()

# %%
# Then the small:
truth = Wavefront(focal_image(2, amp_small).electric_field + focal_image(3, amp_small).electric_field)
recon = prop.backward(truth)
imshow_field(recon.phase * aperture, mask=aperture)
plt.title(f"Zernike mode sums with amplitude = {amp_small}")
plt.colorbar()

# %%