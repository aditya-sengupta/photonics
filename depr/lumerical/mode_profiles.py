# %%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from photonics import read_csv_and_split_tables, LumericalData, PROJECT_ROOT
from scipy.interpolate import griddata
from os import path
from tqdm import trange

njacket, ncladding, ncore = 1.44, 1.4504, 1.4592
# %%
mode_dataset = read_csv_and_split_tables("../../data/lumerical/mode_profiles_231011.txt")
frequency, index_loss, xy, z, index_modes = mode_dataset
# %%
lum = LumericalData(xy)
lum.compute_grid(2e-8)
lum.plot(index_modes['index_x'])
plt.title("Index of refraction profile")

# %%
ax = plt.gca()
mode_r = np.arange(len(index_loss))
ax.fill_between((mode_r - 1) * 1.05, ncladding, ncore, alpha=0.2, label="Guided modes")
ax.fill_between((mode_r - 1) * 1.05, njacket, ncladding, alpha=0.2, label="Radiating modes")
plt.scatter(mode_r, index_loss["effective_index(real)"], s=1, label="Lumerical FDE modes")
plt.xlabel("Mode number")
plt.ylabel("Effective refractive index")
plt.legend()
plt.title("The 7-port lantern has 14 guided supermodes at the SEAL wavelength")
plt.savefig(path.join(PROJECT_ROOT, "figures", "supermodes.png"))

# %%
nrows = 3
N = 14
ncols = int(np.ceil(N / nrows))
figx = 15
fig, axs = plt.subplots(nrows, ncols, figsize=(15, figx * nrows / ncols), sharex=True, sharey=True)
for k in trange(N):
    ax = axs[k // ncols, k % ncols]
    ax.pcolormesh(*lum.regrid(index_modes[f"mode{k+1}_Ex(real)"] ** 2 + index_modes[f"mode{k+1}_Ex(imag)"] ** 2), cmap="gray")
    ax.contour(lum.x, lum.y, lum.r, [8e-6], colors="white")
    ax.set_title(f"Mode {k+1}")
for k in range(N, nrows * ncols):
    fig.delaxes(axs[k // ncols, k % ncols])
plt.suptitle("Guided modes of a 7-port photonic lantern")
plt.savefig(path.join(PROJECT_ROOT, "figures", "pl_guided.png"))
# %% plot all the radiating modes
nrows = 4
Nrad = 28
ncols = int(np.ceil(Nrad / nrows))
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8), sharex=True, sharey=True)
for k in trange(Nrad):
    ax = axs[k // ncols, k % ncols]
    ax.pcolormesh(*lum.regrid(index_modes[f"mode{N+k+1}_Ex(real)"] ** 2 + index_modes[f"mode{N+k+1}_Ex(imag)"] ** 2), cmap="gray")
    ax.contour(lum.x, lum.y, lum.r, [8e-6], colors="white")
for k in range(N + Nrad, N + nrows * ncols):
    fig.delaxes(axs[k // ncols, k % ncols])
plt.suptitle("Radiating modes of the 7-port PL")
plt.savefig(path.join(PROJECT_ROOT, "figures", "pl_radiating.png"))
# %%
