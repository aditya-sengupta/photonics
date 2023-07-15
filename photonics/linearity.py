import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from .utils import *

def make_linearity(dm, cam, reader, Nmodes, lims=(-1.5,1.5), step=0.02, dark=0):
    amp_range = np.arange(lims[0], lims[1]+np.finfo(float).eps, step)
    intensities = np.zeros((Nmodes, len(amp_range), reader.nports))
    for i in trange(2, Nmodes+2):
        for (j, amp) in enumerate(amp_range):
            dm.pokeZernike(amp, i)
            intensities[i-2,j] = reader.get_intensities(cam.get() - dark)
    
    np.save(make_fname("linearity_amplitudes"), amp_range)
    np.save(make_fname("linearity_intensities"), intensities)
    return amp_range, intensities
