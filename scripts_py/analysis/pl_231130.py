"""
This script saves amplitudes and intensities for scripts_jl/interaction_matrix_231130.jl to plot linearity curves.

To be deprecated when I call one library into the other and I can do this data manipulation fully in Julia. Or just ignored.
"""

# %%
import os
import numpy as np
from photonics import LanternReader
from os import path
from tqdm import tqdm

reader = LanternReader(
    18, 10, 3, "npy", (319, 311), subdir="pl_231130"
)
def load(x):
    return np.load(path.join(reader.directory, x))

def normalize(x):
    return x / np.sum(x)
xc, yc = load("xc.npy"), load("yc.npy")
reader.xc = xc - 260
reader.yc = yc - 102

amps = np.arange(-1, 1.05, 0.05)
dmc_files = list(filter(lambda x: x.startswith("dmc_231130_19") or x.startswith("dmc_231130_20"), os.listdir(reader.directory)))
img_files = list(map(lambda x: x.replace("dmc_231130", "pl_231130"), dmc_files))
intensities = []
# DMCs are all saving 0! why??

for imgf in tqdm(np.sort(img_files)[12:24]):
    img = load(imgf)
    intensities.append([normalize(reader.get_intensities(i)) for i in img])

intensities = np.array(intensities)
reader.save("linearity_amps", amps)
reader.save("linearity_intensities", intensities)