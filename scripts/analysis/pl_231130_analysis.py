# %%
import numpy as np
from matplotlib import pyplot as plt
from photonics import LanternReader
from os import path
from tqdm import tqdm

reader = LanternReader(
    18, 10, 3, "npy", (319, 311), subdir="pl_231130"
)
load = lambda x: np.load(path.join(reader.directory, x))
normalize = lambda x: x / np.sum(x)
xc, yc = load("xc.npy"), load("yc.npy")
reader.xc = xc - 260
reader.yc = yc - 102
# %%
"""amps = np.concatenate((np.arange(0.0, 0.3, 0.05), np.arange(0.3, 1.1, 0.1)))
amps = np.concatenate(((-amps[1:])[::-1], amps))"""
amps = np.arange(-1, 1.05, 0.05)
# %%
dmc_files = list(filter(lambda x: x.startswith("dmc_231130_19") or x.startswith("dmc_231130_20"), os.listdir(reader.directory)))
img_files = list(map(lambda x: x.replace("dmc_231130", "pl_231130"), dmc_files))
# %%
intensities = []
# DMCs are all saving 0! why??

for imgf in tqdm(np.sort(img_files)[12:24]):
    img = load(imgf)
    intensities.append([normalize(reader.get_intensities(i)) for i in img])

intensities = np.array(intensities)
"""# %%
push = intensities[:,21,:]
pull = intensities[:,19,:]
int_mat = np.transpose((push - pull) / (2 * 0.05))"""
# %%
reader.save("linearity_amps", amps)
# %%
reader.save("linearity_intensities", intensities)
# %%
