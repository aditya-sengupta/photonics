# improvising this
# todo later/in consultation with Vincent or after we've agreed on a procedure, put this into libSEAL

from tqdm import trange

from .pl_setup import *

poke_amplitude = 0.1

def pl_interaction_matrix(dm, cam):
    N = reader.nports
    A = np.zeros((N, N))
    for z in trange(reader.nports):
        for s in [-1, 1]:
            dm.pokeZernike(s * amp, z + 2)
            A[z, :] += (s / 2) * reader.get_intensities(cam.get())
    return A

def pl_command_matrix(dm, cam):
    IM = pl_interaction_matrix(dm, cam)
    return np.linalg.pinv(IM, rcond=1e-6)

# probably check if the command matrix evaluated at the flat position is all ~zero

def close_loop(dm, cam, CM, niter=10, gain=0.9, leak=0.99):
    prev_dmc = np.zeros(reader.nports)
    for _ in range(niter):
        measurement = reader.get_intensities(cam.get())
        dmc = gain * (CM @ measurement) + leak * prev_dmc
        prev_dmc = dmc
        dm.pokeZernike(0, 0)
        for (z, d) in enumerate(dmc):
            dm.pokeZernike(z + 2, d, bias=dm.current_surf)
