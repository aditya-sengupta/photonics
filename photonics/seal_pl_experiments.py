import numpy as np

from .utils import *

rms = lambda x: np.sqrt(np.sum(x ** 2))

def slm_onestep(plwfs, slm, dm_slm, lim=0.02, nmodes=3, niter=5, verbose=True):
    dm_slm.pokeZernike(0.0, 1)
    rms_baseline = rms(slm.phasemap)
    mode_numbers = np.arange(2, nmodes + 2)
    if plwfs.cmdMat.shape[0] != nmodes:
        plwfs.calibrate(dm_slm, amp_calib=lim, modes_number=nmodes)
    plwfs.update_flat(dm_slm)
    amp = np.random.uniform(-lim, lim, nmodes)
    dm_slm.pokeZernike(amp, mode_numbers)
    rms_before = rms(slm.phasemap)
    if verbose:
        print(f"Applied        {amp}")
    reconstructions = []
    for _ in range(niter):
        reconstructions.append(plwfs.img2cmd(plwfs.getImage()))
    reconstructions = np.array(reconstructions)
    reconstruction = np.mean(reconstructions, axis=0)
    if verbose:
        print(f"Reconstructed  {reconstruction}")
    if niter > 1:
        sd = np.std(reconstructions, axis=0)
        if verbose:
            print(f"Recon. z-score {(amp - reconstruction) / sd}")
    dm_slm.pokeZernike(amp - reconstruction, mode_numbers)
    rms_after = rms(slm.phasemap)
    return (rms_after - rms_baseline) / (rms_before - rms_baseline)

def pl_correct(plwfs, dm, controller, amp, zern, niter=10):
    plwfs.update_flat(dm)
    true_flat = np.copy(dm.flat_surf)
    dm.newFlat(dm.pokeZernike(amp, zern))
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Done, press Enter to end. ")
    dm.newFlat(true_flat)

def pl_correct_slmzern(plwfs, dm, controller, slm, dm_slm, niter=10, limslm=0.3):
    plwfs.update_flat(dm)
    # u, s, v = np.linalg.svd(plwfs.cmdMat)
    z = np.arange(2, 5)
    amp = np.random.uniform(-limslm, limslm, len(z))
    # u @ np.linalg.lstsq(u, amp, rcond=-1)[0], 
    dm_slm.pokeZernike(amp, z)
    init_wf_read = plwfs.img2cmd(plwfs.getImage())
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()) / init_wf_read)
    input("Done, press Enter to end.")
    slm.reset()
    # dm_slm.pokeZernike(0.0, 1)
    dm.pokeZernike(0.0, 1)

# pl_correct_slmzern(dm_mems, controller, dm_slm)

def pl_turb_correct(plwfs, dm, controller, turb, slm, niter=10):
    plwfs.update_flat(dm)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Done, press Enter to end.")
    slm.reset()
    dm.pokeZernike(0.0, 1)
