import numpy as np
import sys
from photonics import LanternReader, date_now
from photonics import slm_onestep, pl_controller, pl_correct, pl_correct_slmzern, pl_turb_correct

sys.path.append(r"/home/lab/libSEAL")
from wfAffectors import SLM, DM_seal
from wfSensors import PhotonicLantern
from cameras import blackFly_camera
from wfControllers import integrator
from tools import rms

reader = LanternReader(
    nports = 18,
    fwhm = 18,
    threshold = 25,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_" + date_now()
)

slm = SLM()
dm_slm = DM_seal('slm',slm,[5,0.5])

c = blackFly_camera("PhotoL.sh")
input("Taking a dark frame, turn off the laser.")
c.getDark()
input("Turn on the laser and adjust power until most ports are not saturated.")
plwfs = PhotonicLantern(c, reader)

# turb = atm(1.5, 635e-9, 0.75, 5, 40, 1e-3, slm)
