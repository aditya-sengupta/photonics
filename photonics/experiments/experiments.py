import os
from os import path
import subprocess
import numpy as np
from itertools import product
from functools import reduce
from time import sleep, time
from tqdm import tqdm, trange
import warnings
import h5py
import paramiko

from .deformable_mirrors import ShaneDM, IrisDM, SimulatedDM
from .lantern_cameras import Goldeye, SimulatedLanternCamera
from ..utils import date_now, datetime_now, datetime_ms_now, time_ms_now, rms, DATA_PATH, zernike_names, center_of_mass, normalize

class Experiments:
    def __init__(self, dm, wfs):
        self.dm = dm
        self.wfs = wfs
        subdir = f"pl_{date_now()}"
        self.subdir = subdir
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        self.ext = "npy"
        self.pl_flat = None
        self.save_full_frame = False

        print(f"Path for data saving set to {self.directory}")
        
    @property
    def directory(self):
        return path.join(DATA_PATH, self.subdir)

    def filepath(self, fname, ext=None):
        """
        The path we want to save/load data from or to.
        """
        if ext is None:
            ext = self.ext
        return path.join(DATA_PATH, self.subdir, fname + "." + ext)
        
    def save(self, fname, data, ext="npy", verbose=True):
        fpath = self.filepath(fname, ext=ext)
        if verbose:
            print(f"Saving data to {fpath}")
        np.save(self.filepath(fname, ext=ext), data)
        
    # deleted utilities for saturation and ROI; reimplement/pull from Git when needed.
  
    def measure_pl_flat(self):
        self.dm.send_zeros(verbose=True)
        self.pl_flat = self.wfs.get_intensities(self.wfs.get_image())
        self.save(f"pl_flat_{datetime_now()}", self.pl_flat)

    def openloop_experiment(self, patterns, tag="openloop_experiment", delay=0):
        start_stamp = datetime_ms_now()
        self.send_zeros(verbose=False)
        img = self.get_image()
        start_time_stamp = time_ms_now()
        bbox_shape = self.get_image().shape
        pl_readout = np.zeros((len(patterns), *bbox_shape))
        
        for (i, p) in enumerate(tqdm(patterns)):
            self.curr_dmc = p
            self.command_to_dm(verbose=False)
            sleep(delay)
            img = self.get_image()
            pl_readout[i] = img
        
        pl_intensities = np.array(list(map(self.get_intensities, pl_readout)))
        self.send_zeros(verbose=False)
        with h5py.File(self.filepath(f"{tag}_{datetime_now()}", ext="hdf5"), "w") as f:
            dmcs_dataset = f.create_dataset("dmc", data=patterns)
            if self.save_full_frame:
                f.create_dataset("pl_images", data=pl_readout)
            pl_intensities_dataset = f.create_dataset("pl_intensities", data=pl_intensities)
            pl_intensities_dataset.attrs["exp_ms"] = self.exp_ms
            pl_intensities_dataset.attrs["gain"] = self.gain
            pl_intensities_dataset.attrs["nframes"] = self.nframes
            pl_intensities_dataset.attrs["centroids_dt"] = self.centroids_dt

        self.destroy_paramiko()
        return pl_intensities

    def save_current_image(self, tag=""):
        self.save(f"pl_image_{datetime_now()}_{tag}", self.get_image())

    def make_interaction_matrix(self, amp_calib=0.01, thres=1/30, nm=None):
        self.setup_paramiko()
        if nm is None:
            nm = self.Nmodes
        self.int_mat = np.zeros((self.Nports, nm))
        pushes = []
        pulls = []
        for i in trange(nm):
            self.zern_to_dm(i + 1, amp_calib, verbose=False)
            sleep(0.1)
            s_push = self.get_image()
            pushes.append(s_push)
            self.zern_to_dm(i + 1, -amp_calib, verbose=False)
            sleep(0.1)
            s_pull = self.get_image()
            pulls.append(s_pull)
            s = (self.get_intensities(s_push) - self.get_intensities(s_pull)) / (2 * amp_calib)
            self.int_mat[:,i] = s.ravel()
   
        self.cmd_mat = np.linalg.pinv(self.int_mat, thres)
        self.cmd_dt = datetime_now()
        with h5py.File(self.filepath(f"intcmd_{self.cmd_dt}", ext="hdf5"), "w") as f:
            pushes_dset = f.create_dataset("push_frames", data=np.array(pushes))
            pulls_dset = f.create_dataset("pull_frames", data=np.array(pulls))
            intmat_dset = f.create_dataset("intmat", data=self.int_mat)
            intmat_dset.attrs["amp_calib"] = amp_calib
            intmat_dset.attrs["exp_ms"] = self.exp_ms
            intmat_dset.attrs["gain"] = self.gain
            intmat_dset.attrs["nframes"] = self.nframes
            cmdmat_dset = f.create_dataset("cmdmat", data=self.cmd_mat)
            cmdmat_dset.attrs["thres"] = thres
   
        self.send_zeros()
        self.destroy_paramiko()
        
    def load_interaction_matrix(self, fname):
        with h5py.File(fname) as f:
            self.exp_ms = f["intmat"].attrs["exp_ms"]
            self.gain = f["intmat"].attrs["gain"]
            self.int_mat = np.array(f["intmat"])
            self.cmd_mat = np.array(f["cmdmat"])

    def pseudo_cl_iteration(self, gain=0.1, verbose=True):
        dm_start = np.copy(self.curr_dmc)
        if verbose:
            print(f"Initial error = {rms(self.curr_dmc)}")
        lantern_reading = np.zeros(self.Nmodes)
        lantern_image = self.get_image()
        lr = self.cmd_mat @ (self.get_intensities(lantern_image) - self.pl_flat)
        lantern_reading[:len(lr)] = lr
        # does the sign of measured WF errors match the actual signs?
        sign_match = ''.join(map(lambda x: str(int(x)), (np.sign(lantern_reading * self.curr_dmc) * 2 + 2)/4))
        if verbose:
            print(f"{sign_match}")
        self.curr_dmc -= gain * lantern_reading
        self.command_to_dm(verbose=verbose)
        dm_end = np.copy(self.curr_dmc)
        if verbose:
            print(f"Final error = {rms(self.curr_dmc)}")
        
        if not verbose:
            return lantern_image, lantern_reading

    def closed_loop(self, gain=0.1, niter=20):
        dmcs = [np.copy(self.curr_dmc)]
        lantern_images, lantern_readings = [], []
        for i in trange(niter):
           lantern_image, lantern_reading = self.pseudo_cl_iteration(gain=gain, verbose=False)
           dmcs.append(np.copy(self.curr_dmc))
           lantern_images.append(lantern_image)
           lantern_readings.append(lantern_reading)
           
        with h5py.File(self.filepath(f"closedloop_{datetime_now()}", ext="hdf5"), "w") as f:
            dmcs_dset = f.create_dataset("dmcs", data=np.array(dmcs))
            dmcs_dset.attrs["exp_ms"] = self.exp_ms
            dmcs_dset.attrs["gain"] = self.gain
            dmcs_dset.attrs["centroids_timestamp"] = self.centroids_dt
            dmcs_dset.attrs["cmd_timestamp"] = self.cmd_dt
            images_dset = f.create_dataset("lantern_images", data=np.array(lantern_images))
            images_dset.attrs["spot_radius_px"] = self.spot_radius_px
            f.create_dataset("lantern_readings", data=np.array(lantern_readings))

        return dmcs, lantern_readings
    
    def closed_loop_replay(self, fname):
        with h5py.File(fname) as f:
            dmcs = np.array(f["dmcs"])
        for (i, dmc) in enumerate(dmcs):
            self.curr_dmc = dmc
            self.command_to_dm()
            # I don't have programmatic access to the PSF camera, so I'll just wait for user input
            input(f"Frame {i}.")

    def measure_linearity(self, min_amp=-0.06, max_amp=0.06, step=0.01, nmodes=None):
        if nmodes is None:
            nmodes = self.Nmodes
        all_recon = []
        amps = None
        for z in range(1, nmodes+1):
            print(f"probing {zernike_names[z+1]}")
            amps, responses = self.sweep_mode(z, min_amp=min_amp, max_amp=max_amp, step=step)
            reconstructed = [self.cmd_mat @ (r - self.pl_flat) for r in responses]
            all_recon.append(reconstructed)

        all_recon = np.array(all_recon)
        with h5py.File(self.filepath(f"linearity_{datetime_now()}", ext="hdf5"), "w") as f:
            f.create_dataset("amps", data=amps)
            all_recon_dset = f.create_dataset("all_recon", data=all_recon)
            all_recon_dset.attrs["cmd_timestamp"] = self.cmd_dt
        return amps, all_recon

    # different types of experiment
    def sweep_mode(self, z, min_amp=-1.0, max_amp=1.0, step=0.1):
        amps = np.arange(min_amp, max_amp+2*step, step)
        patterns = np.zeros((len(amps), self.Nmodes))
        patterns[:,z-1] = amps
        return amps, self.openloop_experiment(patterns, tag=f"sweep_mode_{z}")

    def sweep_all_modes(self, min_amp=-1.0, max_amp=1.0, step=0.1):
        amps = np.arange(min_amp, max_amp+step, step)
        patterns = np.zeros((len(amps) * self.Nmodes + 1, self.Nmodes))
        for z in range(1, self.Nmodes+1):
            patterns[len(amps)*(z-1):len(amps)*(z),z-1] = amps
        return amps, self.openloop_experiment(patterns, tag="sweep_all_modes")

    def random_combinations(self, Niters, lim=0.05):
        inputzs = np.random.uniform(-lim, lim, (Niters+1, self.Nmodes))
        self.openloop_experiment(inputzs, tag="random_combinations")

    def exptime_sweep(self, exp_ms_values, tag=""):
        sweep_values = []
        for exp_ms in exp_ms_values:
            self.exp_ms = exp_ms
            sweep_values.append(self.get_intensities(self.get_image()))
        
        sweep_values = np.array(sweep_values)
        with h5py.File(self.filepath(f"{tag}_sweep_{datetime_now()}", ext="hdf5"), "w") as f:
            exp_ms_dset = f.create_dataset("exp_ms", data=exp_ms_values)
            exp_ms_dset.attrs["gain"] = self.gain
            img_dset = f.create_dataset(f"{tag}_sweep_readout", data=sweep_values)
            img_dset.attrs["nframes"] = self.nframes
            img_dset.attrs["dark_subtracted"] = "false"

        return sweep_values

    def snr_per_port(self, exp_ms_low, exp_ms_high, exp_ms_step):
        orig_dark = np.copy(self.dark)
        self.dark = 0
        exp_ms_values = np.arange(exp_ms_low, exp_ms_high + exp_ms_step, exp_ms_step)
        input("Remove light, taking dark frames")
        dark_sweep_values = self.exptime_sweep(exp_ms_values, tag="dark")
        input("Restore light, taking PL images")
        lantern_sweep_values = self.exptime_sweep(exp_ms_values, tag="lantern")
        snr = (lantern_sweep_values - dark_sweep_values) / np.sqrt(lantern_sweep_values)
        self.dark = orig_dark
        return exp_ms_values, snr
    
    def stability(self, n_pictures=20, wait_s=10):
        patterns = np.tile(self.curr_dmc, (n_pictures,1))
        self.openloop_experiment(patterns, tag="stability", delay=wait_s)
        
    def stability_better(self, n_pictures=20, wait_s=2):
        start_stamp = datetime_ms_now()
        img = self.get_image()
        start_time_stamp = time_ms_now()
        bbox_shape = self.get_image().shape
        pl_readout = np.zeros((n_pictures, *bbox_shape))
        for i in trange(n_pictures):
            sleep(wait_s)
            img = self.get_image()
            pl_readout[i] = img
        
        pl_intensities = np.array(list(map(self.get_intensities, pl_readout)))
        with h5py.File(self.filepath(f"stability_{datetime_now()}", ext="hdf5"), "w") as f:
            if self.save_full_frame:
                f.create_dataset("pl_images", data=pl_readout)
            pl_intensities_dataset = f.create_dataset("pl_intensities", data=pl_intensities)
            pl_intensities_dataset.attrs["exp_ms"] = self.exp_ms
            pl_intensities_dataset.attrs["gain"] = self.gain
            pl_intensities_dataset.attrs["nframes"] = self.nframes
            pl_intensities_dataset.attrs["centroids_dt"] = self.centroids_dt

        return pl_intensities
        
    def focus_astig_cube_sweep(self):
        # NOT writing this to generalize to higher orders or further out because dear god no
        patterns = np.zeros((13**3, self.Nmodes))
        amps_per_mode = np.arange(-0.06, 0.07, 0.01)
        for (i, pattern) in enumerate(product(amps_per_mode, amps_per_mode, amps_per_mode)):
            patterns[i,:3] = pattern
            
        self.openloop_experiment(patterns, tag="focus_astig_cube_sweep", delay=0.3)
            