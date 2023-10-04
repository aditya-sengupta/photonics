# %%
import hcipy
from hcipy import mode_basis, make_fft_grid, imshow_field
# %%
pupil_grid = make_pupil_grid(256, 1)
z_mb = mode_basis.make_zernike_basis(18, 1, pupil_grid)
f_grid = make_fft_grid(pupil_grid)
f_mb = mode_basis.make_fourier_basis(pupil_grid, f_grid)
# %%
imshow_field(f_mb[20])
# %%
len(f_mb)
# %%
