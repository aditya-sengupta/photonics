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
rms = lambda x: np.sqrt(np.sum(x ** 2))

# turb = atm(1.5, 635e-9, 0.75, 5, 40, 1e-3, slm)

def slm_onestep(dm_slm, lim=0.02, nmodes=3):
    mode_numbers = np.arange(2, nmodes + 2)
    if plwfs.cmdMat.shape[0] != nmodes:
        plwfs.calibrate(dm_slm, lim, nmodes)
    plwfs.update_flat(dm_slm)
    amp = np.random.uniform(-lim, lim, nmodes)
    dm_slm.pokeZernike(amp, mode_numbers)
    rms_before = rms(slm.phasemap)
    print(f"Applied        {amp}")
    reconstructions = []
    for _ in range(5):
        reconstructions.append(plwfs.img2cmd(plwfs.getImage()))
    reconstructions = np.array(reconstructions)
    reconstruction = np.mean(reconstructions, axis=0)
    sd = np.std(reconstructions, axis=0)
    print(f"Reconstructed  {reconstruction}")
    print(f"Recon. z-score {(amp - reconstruction) / sd}")

    dm_slm.pokeZernike(amp - reconstruction, mode_numbers)
    rms_after = rms(slm.phasemap)
    return rms_after / rms_before

def pl_controller(dm, gain=0.1):
    controller = integrator(plwfs, dm)
    controller.loopgain = gain
    return controller

def pl_correct(dm, controller, amp, zern, niter=10):
    plwfs.update_flat(dm)
    true_flat = np.copy(dm.flat_surf)
    dm.newFlat(dm.pokeZernike(amp, zern))
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Done, press Enter to end. ")
    dm.newFlat(true_flat)

def pl_correct_slmzern(dm, controller, dm_slm, niter=10, limslm=0.3):
    plwfs.update_flat(dm)
    # u, s, v = np.linalg.svd(plwfs.cmdMat)
    z = np.arange(2, 5)
    amp = np.random.uniform(-limslm, limslm, len(z))
    dm_slm.pokeZernike(
        amp,
        #u @ np.linalg.lstsq(u, amp, rcond=-1)[0], 
        z
    )
    init_wf_read = plwfs.img2cmd(plwfs.getImage())
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()) / init_wf_read)
    input("Done, press Enter to end.")
    slm.reset()
    # dm_slm.pokeZernike(0.0, 1)
    dm.pokeZernike(0.0, 1)

# pl_correct_slmzern(dm_mems, controller, dm_slm)

def pl_turb_correct(dm, controller, turb, slm, niter=10):
    plwfs.update_flat(dm)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Done, press Enter to end.")
    slm.reset()
    dm.pokeZernike(0.0, 1)

#print("Initialized camera, setting port positions")
#reader.set_centroids(c.get(100))

"""input("Turn off the laser!")
dark = c.get(100)
input("Turn on the laser!")

# stop here, check port masking etc
# then do run(dm=dm_alpao) or run(dm=dm_alpao, N=1000)

run = partial(random_testing, dm=dm_slm, cam=c, reader=reader, lims=(-0.12,0.12), modes=np.arange(2, 11))"""
