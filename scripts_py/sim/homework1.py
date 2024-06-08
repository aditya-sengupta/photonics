# %%
import numpy as np
import matplotlib.pyplot as plt
from hcipy import get_strehl_from_focal
from tqdm import trange
from photonics.simulations.optics import Optics
from photonics.simulations.pyramid_optics import PyramidOptics

optics = Optics(lantern_fnumber=6.5, dm_basis="modal")
pyramid = PyramidOptics(optics)
gain = 0.3
leakage = 0.999
dt=1./800

num_iterations = 200 #number of time steps in our simulation. We'll run for a total of dt*num_iterations seconds
sr = [] # so we can find the average strehl ratio
wavefronts_after_dm = []

optics.layer.reset()
optics.layer.t = 0
for timestep in trange(num_iterations):
    wf_after_dm = optics.wavefront_after_dm(timestep * dt)
    optics.deformable_mirror.actuators = leakage*optics.deformable_mirror.actuators - gain * pyramid.reconstruct(wf_after_dm)
    wf_focal = optics.focal_propagator.forward(wf_after_dm)
    wavefronts_after_dm.append(wf_after_dm)
    strehl_foc = get_strehl_from_focal(wf_focal.intensity/optics.norm,optics.im_ref.intensity/optics.norm)
    sr.append(strehl_foc)

plt.plot(sr)
# %%
