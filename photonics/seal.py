import numpy as np
from scipy.stats import linregress
from time import sleep
from tqdm import tqdm, trange

from .utils import *

rms = lambda x: np.sqrt(np.sum(x ** 2))

def reconstruction_reliability(plwfs, slm, dm_slm, lim=0.05, nmodes=10, ntry=5):
    slm.reset()
    mode_numbers = np.arange(2, nmodes + 2)
    if plwfs.cmdMat.shape[0] != nmodes:
        plwfs.calibrate(dm_slm, amp_calib=lim, modes_number=nmodes)
    plwfs.update_flat(dm_slm)
    amp = np.random.uniform(-lim, lim, nmodes)
    dm_slm.pokeZernike(amp, mode_numbers)
    reconstructions = []
    for _ in trange(ntry):
        reconstructions.append(plwfs.img2cmd(plwfs.getImage()))
    sd = np.std(reconstructions, axis=0)
    avg = np.mean(reconstructions, axis=0)
    print(f"Recon. z-score {(amp - avg) / sd}")
    slm.reset()

def closeloop_slm(plwfs, slm, dm_slm, amp=None, gain=0.1, lim=0.05, nmodes=10, niter=10, verbose=True):
    dm_slm.pokeZernike(0.0, 1)
    rms_baseline = rms(slm.phasemap)
    mode_numbers = np.arange(2, nmodes + 2)
    if plwfs.cmdMat.shape[0] != nmodes:
        plwfs.calibrate(dm_slm, amp_calib=lim, modes_number=nmodes)
    plwfs.update_flat(dm_slm)
    if amp is None:
        amp = np.random.uniform(-lim, lim, nmodes)
    curr_slm = np.copy(amp)
    dm_slm.pokeZernike(amp, mode_numbers)
    phasemap_before = np.copy(slm.phasemap)
    rms_before = rms(phasemap_before)
    init_lant_rms = rms(plwfs.img2cmd(plwfs.getImage()))
    recons = []
    for _ in trange(niter):
        reconstruction = plwfs.img2cmd(plwfs.getImage())
        recons.append(reconstruction)
        curr_slm = curr_slm - gain * reconstruction
        dm_slm.pokeZernike(curr_slm, mode_numbers)
    rms_after = rms(slm.phasemap)
    improvement = (rms_after - rms_baseline) / (rms_before - rms_baseline)
    lant_improvement = rms(plwfs.img2cmd(plwfs.getImage())) / init_lant_rms
    if verbose:
        print(f"SLM improvement     {improvement}")
        print(f"Lantern improvement {lant_improvement}")
    return amp, recons, curr_slm, improvement

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

def exposure_snr_testing(plwfs, cam, tmin=5, tmax=75):
    times = np.arange(tmin, tmax)
    darks = []
    input("Sweeping through exposure times to take dark frames - turn off the laser.")
    for t in tqdm(times):
        cam.set_exp(t, print_output=False)
        darks.append(cam.get() + cam.dark)
    input("Done taking dark frames, turn on the laser.")
    averages, peaks = [], []
    for (t, d) in zip(tqdm(times), darks):
        cam.set_exp(t, print_output=False)
        averages.append(reader.get_intensities(cam.get() + cam.dark - d))
        peaks.append(reader.peaks_per_port(cam.get() + cam.dark))
    return times, darks, averages, peaks

def monitor_saturation(plwfs, cam, n=100, dt=5):
    peaks = []
    for _ in trange(n):
        peaks.append(reader.peaks_per_port(cam.get() + cam.dark))
        sleep(dt)

    return peaks

def monitor_saturation_slm(plwfs, cam, dm_slm, lim=0.3, n=20, dt=0, niter=1):
    peaks = []
    z = np.arange(2, 19)
    for _ in trange(n):
        dm_slm.pokeZernike(np.random.uniform(-lim, lim, len(z)), z)
        for _ in range(niter):
            peaks.append(reader.peaks_per_port(cam.get() + cam.dark))
            sleep(dt)

    return peaks

def monitor_stability(plwfs, cam, n=100):
    measurements = []
    norm = np.sum(reader.port_mask()) / reader.nports
    for _ in trange(n):
        measurements.append(reader.get_intensities(cam.get()) / norm)

    return measurements

def snr_per_port(plwfs):
    reader = plwfs.reader
    img = plwfs.cam.get()
    rp = 2 * (reader.fwhm + 1)
    comparable_xc = np.arange(rp, rp * (reader.nports + 1), rp)
    comparable_yc = rp * np.ones(reader.nports)
    lantern_readings = reader.get_intensities(img)
    comparable_readings = np.zeros(reader.nports)
    comparable_stdevs = np.zeros(reader.nports)
    for (i, (xp, yp)) in enumerate(zip(comparable_xc, comparable_yc)):
        masked = img[np.where((reader.xi - xp) ** 2 + (reader.yi - yp) ** 2 <= reader.fwhm ** 2)]
        comparable_readings[i] = np.sum(masked)
        comparable_stdevs[i] = np.std(masked)
    signal = (lantern_readings - comparable_readings) / (np.sum(reader.port_mask()) / reader.nports)
    noise = np.sqrt(signal + comparable_stdevs ** 2)
    return signal / noise

def max_counts_per_aperture(reader, img):
    r = reader.fwhm
    max_vals = []
    for (x, y) in zip(reader.xc, reader.yc):
        lower_x, upper_x = int(np.floor(x - r)), int(np.ceil(x + r))
        lower_y, upper_y = int(np.floor(y - r)), int(np.ceil(y + r))
        masked = img[
            lower_y:upper_y,
            lower_x:upper_x
        ]
        max_vals.append(np.max(masked))

    return max_vals