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

def pl_correct_slmzern(dm, controller, dm_slm, niter=10, limslm=0.3, project=False):
    plwfs.update_flat(dm)
    if project:
        u, s, v = np.linalg.svd(plwfs.intMat)
        amp = u @ np.linalg.lstsq(u, amp, rcond=-1)[0]
    # u, s, v = np.linalg.svd(plwfs.cmdMat)
    amp = np.random.uniform(-limslm, limslm, plwfs.reader.nports)
    dm_slm.pokeZernike(
        amp,
        np.arange(2, 20)
    )
    init_wf_read = plwfs.img2cmd(plwfs.getImage())
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()) / init_wf_read)
    input("Done, press Enter to end.")
    slm.reset()
    # dm_slm.pokeZernike(0.0, 1)
    dm.pokeZernike(0.0, 1)

def pl_turb_correct(dm, controller, turb, slm, niter=10):
    plwfs.update_flat(dm)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Start loop closing.")
    controller.closeLoop(niter)
    print(plwfs.img2cmd(plwfs.getImage()))
    input("Done, press Enter to end.")
    slm.reset()
    dm.pokeZernike(0.0, 1)
