import numpy as np
from tqdm import tqdm

from ..utils import datetime_now, date_now

datapath = f"/home/lab/asengupta/photonics/data/pl_{date_now()}"

def sharpen_psf(dm, controller, z2=3, z3=3):
    """
    Steers the PSF back to the center, closes the loop, and sends it back to the desired point.
    """
    dm.setFlatSurf()
    controller.closeLoop(20)
    dm.pokeZernike(z2, 2)
    dm.pokeZernike(z3, 3, bias=dm.current_surf)

def sweep_mode(dm, cam, zern, minval=-1, maxval=1.05, step=0.05, device="pl"):
    for amp in np.arange(minval, maxval, step):
        dm.pokeZernike(amp, zern)
        img = cam.get()
        np.save(f"{datapath}/{device}_{datetime_now()}_z{zern}_a{amp}", img)

def random_testing(dm, cam, reader, N=3, lims=(-0.5,0.5), modes=np.arange(2, 20), dark=0):
    stamp = datetime_now()
    input_wavefronts = np.random.uniform(low=lims[0], high=lims[1], size=(N,len(modes)))
    np.save(f"{datapath}/inputzs_{stamp}", input_wavefronts)
    # imgshape = cam.get(1).shape
    intensities = np.zeros((N,reader.nports))
    for (i, input_wf) in enumerate(tqdm(input_wavefronts)):
        dm.setFlatSurf()
        for (z, amp) in zip(modes, input_wf):
            dm.pokeZernike(amp, z, bias=dm.current_surf)
        img = cam.get()
        intensities[i,:] = reader.get_intensities(img - dark)

    np.save(f"{datapath}/pls_{stamp}", intensities)
