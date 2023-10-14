import numpy as np
import sys
from photonics import LanternReader, date_now, pl_controller, pl_correct, pl_correct_slmzern, pl_turb_correct

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

# turb = atm(1.5, 635e-9, 0.75, 5, 40, 1e-3, slm)