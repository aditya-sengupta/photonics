# %%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# %%
import matplotlib as mpl
# mpl.use('pgf')
import numpy as np
from hcipy import imshow_field
from tqdm import tqdm, trange
from photonics.linearity import plot_linearity
from photonics.simulations.command_matrix import make_command_matrix
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics
# %%
optics = Optics(lantern_fnumber=6.5)
pyramid = PyramidOptics(optics)
dm = optics.deformable_mirror
# %%
make_command_matrix(dm, pyramid, optics.wf)
# %%
dm.actuators[3] = 0.1 * optics.wl / (2 * np.pi)
pyramid.reconstruct(dm.forward(optics.wf))
# %%
zr = np.arange(9)
ar = np.arange(-1.0, 1.01, 0.01)
sweep = np.zeros((len(zr), len(ar), len(zr)))
for (i, z) in enumerate(tqdm(zr)):
    for (j, a) in enumerate(ar):
        dm.actuators[z] = a * optics.wl / (2 * np.pi)
        recon = pyramid.reconstruct(dm.forward(optics.wf))[:len(zr)]
        sweep[i,j,:] = recon * (2 * np.pi) / optics.wl
        
    dm.flatten()
    
# %%
plot_linearity(ar, sweep, "modulated pyramid WFS", savepath="../../figures/mpwfs_linearity.pdf")
# %%
np.save("../../data/linear_sweeps/mpwfs.npy", sweep)
# %%
