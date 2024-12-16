from photonics import SimulatedDM, SimulatedLanternCamera, Experiments
from photonics.simulations.optics import Optics
from hcipy import imshow_field
from matplotlib import pyplot as plt

optics = Optics(lantern_fnumber=10) # so I flood the input/can see all the ports
dm = SimulatedDM(optics.pupil_grid)
pl = SimulatedLanternCamera(dm, optics, "spie24_sim_pl")
exp = Experiments(dm, pl)

pl.measure_dark()
pl.set_centroids(pl.get_image())
