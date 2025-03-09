from seal import Goldeye
from photonics import LanternReader, ShaneDM, Experiments
from photonics.linearity import plot_linearity

camera = Goldeye("test")
lr = LanternReader(camera.get_image(), tag="shane_prep")

dm = ShaneDM()
ex = Experiments(dm, lr, camera)

ex.measure_pl_flat()
ex.save_current_image()
ex.sweep_mode(1)
ex.sweep_all_modes()
ex.random_combinations(30)
ex.measure_linearity(3)
plot_linearity(*ex.measure_linearity(3))
ex.make_interaction_matrix(3)
ex.pseudo_cl_iteration()
