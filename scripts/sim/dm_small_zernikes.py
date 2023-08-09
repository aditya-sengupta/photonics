# %%
from hcipy import *

from fig4_2208_config import *

# %%
wl = 0.65e-6 # m
ampl = 0.0
zern = 3
phase = zernike_ansi(zern, D=D)(pupil_grid)
aberration = np.exp(1j * ampl * phase)
wf = Wavefront(aberration, wavelength=wl)
focal_image = fprop(wf)
plt.imshow(np.log10(focal_image.intensity / focal_image.intensity.max()).shaped[100:-100,100:-100], vmin=-5)
# %%
