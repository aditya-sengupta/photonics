import numpy as np
import sys
from time import sleep
from photonics import LanternReader, date_now, pl_correct, pl_turb_correct

sys.path.append(r"/home/lab/libSEAL")
from wfAffectors import SLM, DM_seal
from wfSensors import PhotonicLantern
from cameras import blackFly_camera
from wfControllers import integrator
from tools import rms

reader = LanternReader(
    nports = 18,
    fwhm = 9,
    threshold = 25,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_" + date_now()
)

c = blackFly_camera("PhotoL.sh")
sleep(3)
input("Adjust power until most ports are not saturated.")
plwfs = PhotonicLantern(c, reader)
plwfs.reader.xc = np.round(plwfs.reader.xc)
plwfs.reader.yc = np.round(plwfs.reader.yc)
plwfs.reader.fwhm = 6

c.nFrames = 100

c.set_exp(40)
input("Taking a dark frame, turn off the laser.")
c.getDark()
input("Turn on the laser.")

def linearity_mems(dm):
    plwfs.update_flat(dm)
    return plwfs.linearity(dm, modes_number=10, amp_calib=0.02, lim=0.05, step=0.01)
