import numpy as np
import sys
from time import sleep
from photonics import LanternReader, date_now, pl_correct, pl_correct_slmzern, pl_turb_correct

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

slm = SLM()
dm_slm = DM_seal('slm',slm,[5,0.5])

c = blackFly_camera("PhotoL.sh")
sleep(3)
input("Taking a dark frame, turn off the laser.")
c.getDark()
input("Turn on the laser and adjust power until most ports are not saturated.")
plwfs = PhotonicLantern(c, reader)
plwfs.reader.xc = np.round(plwfs.reader.xc)
plwfs.reader.yc = np.round(plwfs.reader.yc)
plwfs.reader.fwhm = 9

# turb = atm(1.5, 635e-9, 0.75, 5, 40, 1e-3, slm)
