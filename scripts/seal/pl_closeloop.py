import numpy as np
import sys
from photonics import LanternReader, date_now
from functools import partial
import time
from tqdm import trange

sys.path.append(r"/home/lab/libSEAL")
from wfAffectors import SLM, DM_seal
from wfSensors import PhotonicLantern
from cameras import blackFly_camera
from wfControllers import integrator

reader = LanternReader(
    nports = 18,
    fwhm = 18,
    threshold = 25,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_" + date_now()
)

amp_calib = 0.02
thres = 1/30

slm = SLM()
dm_slm = DM_seal('slm',slm,[5,0.5])

c = blackFly_camera("PhotoL.sh")
plwfs = PhotonicLantern(c, reader)
plwfs.modal = 'zernike'
input("Turn off the laser!")
c.getDark()
input("Turn on the laser!")

calibrate = lambda dm: plwfs.calibrate(dm, amp_calib, reader.nports)

def pl_controller(dm, gain=0.1):
    controller = integrator(plwfs, dm)
    controller.loopgain = gain
    return controller

def pl_correct(dm, controller, amp, zern):
    true_flat = np.copy(dm.flat_surf())
    dm.newFlat(dm.pokeZernike(amp, zern))
    controller.closeLoop(10)
    input("Done, press any key to end. ")
    dm.setFlatSurf(true_flat)


#print("Initialized camera, setting port positions")
#reader.set_centroids(c.get(100))

"""input("Turn off the laser!")
dark = c.get(100)
input("Turn on the laser!")

# stop here, check port masking etc
# then do run(dm=dm_alpao) or run(dm=dm_alpao, N=1000)

run = partial(random_testing, dm=dm_slm, cam=c, reader=reader, lims=(-0.12,0.12), modes=np.arange(2, 11))"""
