# %%
# ask about adding this to default profile on threadripper
%load_ext autoreload
%autoreload 2
# %%
import numpy as np
from matplotlib import pyplot as plt
from photonics.second_stage_optics import SecondStageOptics
# %%
sso = SecondStageOptics()
# %%
psfs = sso.pyramid_correction(gain=0.5)
# %%
for lantern_output in map(lambda p: sso.lantern_optics.lantern_output_to_plot(p), psfs):
    plt.imshow(np.abs(lantern_output) ** 2)
    plt.show()
# %%
