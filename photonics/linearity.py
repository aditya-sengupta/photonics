import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

from .utils import *

def make_linearity(dm, plwfs, Nmodes, lims=(-1.5,1.5), step=0.02):
    amp_range = np.arange(lims[0], lims[1]+np.finfo(float).eps, step)
    intensities = np.zeros((Nmodes, len(amp_range), plwfs.reader.nports))
    for i in range(2, Nmodes+2):
        for (j, amp) in enumerate(tqdm(amp_range)):
            dm.pokeZernike(amp, i)
            intensities[i-2,j] = plwfs.getImage()
    
    np.save(make_fname("linearity_amplitudes"), amp_range)
    np.save(make_fname("linearity_intensities"), intensities)
    return amp_range, intensities

def plot_linearity(plwfs, amp_range, intensities):
    nrows = 3
    N = plwfs.reader.nports
    fig, axs = plt.subplots(nrows, int(np.ceil(plwfs.reader.nports / nrows)), sharex=True, sharey=True)
    for k in range(N):
        axs[k // nrows, k % nrows].plot(amp_range, intensities[k,:,:] @ plwfs.cmdMat)
    plt.show()