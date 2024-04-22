# %%
import numpy as np
from matplotlib import pyplot as plt
import hcipy as hc
from hcipy import imshow_field
from photonics.lantern_optics import LanternOptics
# %%
lo = LanternOptics(f_number=5)
# %%
f_number = 5
spatial_resolution = lo.wl * 1e-6 * f_number / lo.telescope_diameter
q = spatial_resolution / lo.mesh.ds / 1e-6 # pixels per resolution element
num_airy = np.floor(lo.mesh.xw * 1e-6 / (2 * spatial_resolution)) + 1/2 # number of resolution elements
grid = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
# %%
