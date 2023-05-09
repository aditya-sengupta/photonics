# %%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import SiO2
# define geometry

cell = mp.Vector3(30,7,0)

geometry = [mp.Block(mp.Vector3(4,.5,mp.inf),
                     center=mp.Vector3(-13,0,0),
                     material=mp.Medium(epsilon=3.2)),
            mp.Block(mp.Vector3(28,4,mp.inf),
                     center=mp.Vector3(3,0,0),
                     material=mp.Medium(epsilon=3.2))]


# sources = [mp.Source(mp.ContinuousSource(wavelength=1.55),
#                      component=mp.Ez,
#                      center=mp.Vector3(-14,0))]
rot_angle = 0
kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)
bnum = 1
w = .5

sources = [mp.EigenModeSource(src=mp.ContinuousSource(wavelength=1.1),
                                  center=mp.Vector3(-14,0),
                                  size=mp.Vector3(y=3*w),
                                  direction=mp.NO_DIRECTION,
                                  eig_kpoint=kpoint,
                                  eig_band=bnum,
                                  eig_parity=mp.EVEN_Y+mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
                                  eig_match_freq=True)]


# define PML
pml_layers = [mp.PML(1,0)]

# define resolution
resolution = 10

# define simulation
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# # run sim
sim.run(until=500)

# plot results
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)
ey_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
I_data = np.sqrt((ez_data**2) + (ey_data**2) + (ex_data**2))
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(I_data.transpose(), interpolation='spline36', cmap='jet', alpha=.6)
plt.axis('off')
plt.show()


# %%
