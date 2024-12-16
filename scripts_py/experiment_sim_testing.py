from hcipy import imshow_field
from matplotlib import pyplot as plt

from photonics import SimulatedDM, SimulatedLanternCamera, Experiments
from photonics.simulations.optics import Optics
from photonics.linearity import plot_linearity

optics = Optics(lantern_fnumber=10) # so I flood the input/can see all the ports
dm = SimulatedDM(optics.pupil_grid)
pl = SimulatedLanternCamera(dm, optics, "spie24_sim_pl")
exp = Experiments(dm, pl)

pl.measure_dark()
img = pl.get_image()
pl.set_centroids(img, rerun=False) # set to True if we want to redo centroiding
exp.measure_pl_flat()
exp.make_interaction_matrix(9)
amps, all_recon = exp.measure_linearity(3)
