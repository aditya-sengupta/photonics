import numpy as np
from datetime import datetime
from tqdm import tqdm, trange

date_now = lambda: datetime.now().strftime('%Y%m%d')[2:]
time_now = lambda: datetime.now().strftime('%H%M')
datetime_now = lambda: date_now() + "_" + time_now()

datapath = "/home/lab/asengupta/photonics/data"

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

def random_testing(dm, cam, N=3, lims=(-0.5,0.5), modes=np.arange(2, 20)):
    stamp = datetime_now()
    input_wavefronts = np.random.uniform(low=lims[0], high=lims[1], size=(N,len(modes)))
    np.save(f"{datapath}/inputzs_{stamp}", input_wavefronts)
    imgshape = cam.get(1).shape
    camera_outputs = np.zeros((N,imgshape[0],imgshape[1]))
    for (i, input_wf) in enumerate(tqdm(input_wavefronts)):
        dm.setFlatSurf()
        for (z, amp) in zip(modes, input_wf):
            dm.pokeZernike(amp, z, bias=dm.current_surf)
        img = cam.get()
        camera_outputs[i,:,:] = img

    np.save(f"{datapath}/pls_{stamp}", camera_outputs)
