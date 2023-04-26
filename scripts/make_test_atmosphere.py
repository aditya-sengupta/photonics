# %%
import numpy as np
from hcipy import *
from hcipy.atmosphere import *
from tqdm import tqdm
from fig4_2208_config import *

fname = "../data/input_fields_from_test_atm.npy"
# %%
def make_atmosphere():
    r0 = 0.3
    wavelength = 1e-6
    L0 = 10
    velocity = 3
    height = 0
    stencil_length = 2
    oversampling = 128

    mode_basis = make_zernike_basis(500, 1, pupil_grid, 1)

    layers = []
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer2 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer2)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer3 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer3)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer4 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer4)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer5 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer5)
    layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, wavelength), L0, velocity, 0, stencil_length)
    layer6 = ModalAdaptiveOpticsLayer(layer, mode_basis, 1)
    layers.append(layer6)

    atmosphere = MultiLayerAtmosphere(layers, False)
    atmosphere.Cn_squared = Cn_squared_from_fried_parameter(1/40, wavelength)
    return atmosphere

# %%
def make_input_fields(npoints=2001, save=True):
    atmosphere = make_atmosphere()
    aperture = make_circular_aperture(1)(pupil_grid)
    wf = Wavefront(Field(np.ones(pupil_grid.size), pupil_grid), wavelength)
    times = np.linspace(0, 2, npoints)
    input_fields = np.empty((len(times), focal_grid.shape[0], focal_grid.shape[1]), dtype=np.complex128)
    for (i, t) in enumerate(tqdm(times)):
        atmosphere.evolve_until(t)
        wf2 = atmosphere.forward(wf)
        wf2.electric_field *= aperture
        u_in = np.array(fprop(wf2).electric_field).reshape(tuple(focal_grid.shape))
        input_fields[i] = u_in
    if save:
        np.save("../data/input_fields_from_test_atm.npy", input_fields)
    return input_fields

def get_input_fields(*args, **kwargs):
    input_fields = None
    import os
    if not(os.path.isfile(fname)):
        input_fields = make_input_fields(*args, **kwargs)
    else:
        input_fields = np.load(fname)
    return input_fields

# %%
if __name__ == "__main__":
    input_fields = get_input_fields(1, False)


# %%
