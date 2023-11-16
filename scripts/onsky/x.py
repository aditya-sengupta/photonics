# %%
import numpy as np
from matplotlib import pyplot as plt
import PIL
import os
# %%
datapath = "../../data/pl_231102/"
# %%
np.sort(list(filter(lambda x: x.startswith("mode_sweep_all"), os.listdir(datapath))))
# %%
