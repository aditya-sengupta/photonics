# improvising this
# todo later/in consultation with Vincent or after we've agreed on a procedure, put this into libSEAL

from tqdm import trange

from .pl_setup import *

poke_amplitude = 0.1

def pl_interaction_matrix(dm, cam):
    N = reader.nports
    A = np.zeros((N, N))
    for z in trange(reader.nports):
        for amp in [-poke_amplitude, poke_amplitude]:
            dm.pokeZernike(amp, z + 2)
            A[z, :] += reader.get_intensities(cam.get())
    A = A / 2
    return A

def closeloop(gain=0.9, leak=0.99):
    pass