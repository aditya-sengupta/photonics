import numpy as np
import sys
from photonics import LanternReader, sharpen_psf, sweep_mode, random_testing
from functools import partial

sys.path.append(r"/home/lab/libSEAL")
from wfAffectors import SLM, DM_seal
from cameras import blackFly_camera

reader = LanternReader(
    nports = 18,
    fwhm = 18,
    threshold = 25,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_230714"
)

slm = SLM()
nAct_across = 5
c = 0.5
dm_slm = DM_seal('slm',slm,[nAct_across,c])

c = blackFly_camera("PhotoL.sh")
print("Initialized camera, setting port positions")
reader.set_centroids(c.get(10000))

input("Turn off the laser!")
dark = c.get()
input("Turn on the laser!")

# stop here, check port masking etc
# then do run(dm=dm_alpao) or run(dm=dm_alpao, N=1000)

run = partial(random_testing, dm=dm_slm, cam=c, reader=reader, lims=(-0.12,0.12), modes=np.arange(2, 11))
