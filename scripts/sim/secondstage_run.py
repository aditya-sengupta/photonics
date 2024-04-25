# %%
# ask about adding this to default profile on threadripper
%load_ext autoreload
%autoreload 2
# %%
from photonics.second_stage_optics import SecondStageOptics
from photonics.lantern_optics import LanternOptics
# %%
lo = LanternOptics()
# %%
sso = SecondStageOptics(lo)
# %%
sso.pyramid_correction()
# %%
