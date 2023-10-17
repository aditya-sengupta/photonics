import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from .utils import *

def make_linearity(dm, plwfs, Nmodes, lims=(-1.5,1.5), step=0.05):
    amp_range = np.arange(lims[0], lims[1]+np.finfo(float).eps, step)
    intensities = np.zeros((Nmodes, len(amp_range), plwfs.reader.nports))
    for i in range(2, Nmodes+2):
        for (j, amp) in enumerate(tqdm(amp_range)):
            dm.pokeZernike(amp, i)
            intensities[i-2,j] = plwfs.getImage()
    
    np.save(make_fname("linearity_amplitudes"), amp_range)
    np.save(make_fname("linearity_intensities"), intensities)
    return amp_range, intensities

def plot_linearity(plwfs, amp_range, intensities, nrows=3, redo_cmd=False):
    N = intensities.shape[0]
    _, axs = plt.subplots(nrows, int(np.ceil(N / nrows)), sharex=True)

    if redo_cmd:
        push_ref = 0.1
        l, r = np.argmin(np.abs(amp_range + push_ref)), np.argmin(np.abs(amp_range - push_ref))
        intmtx = (intensities[:,r,:] - intensities[:,l,:]) / (2 * push_ref)
        cmd = np.linalg.pinv(intmtx)
    else:
        cmd = plwfs.cmdMat
    for k in range(N):
        commands = intensities[k,:,:] @ cmd
        for j in range(N):
            alpha = 1 if k == j else 0.1
            axs[k % nrows, k // nrows].plot(amp_range, commands[:,j], alpha=alpha)
    plt.show()