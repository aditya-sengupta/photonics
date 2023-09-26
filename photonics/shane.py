import subprocess
import numpy as np
import PySpin
from astropy.io import fits
from functools import reduce
from time import sleep, time
from tqdm import tqdm, trange
import warnings

from .flir import configure_trigger, acquire_image, reset_trigger
from .lantern_reader import LanternReader
from .utils import datetime_ms_now, time_ms_now

def save_telemetry(wait=0):
    warnings.warn("If you see this and you're at Lick, uncommand the line starting with subprocess.run")
    sleep(wait)
    """subprocess.run([
        "ssh", "-Y", 
        "user@shimmy.ucolick.org", "/opt/kroot/bin/modify",
        "-s", "saocon", "savedata=1"
    ])
    """

class ShaneLantern:
    def __init__(self, reader, Nmodes=12):
        self.reader = reader
        self.Nmodes = Nmodes
        self.curr_dmc = np.zeros(Nmodes)
        self.initialize_camera()

    def initialize_camera(self):
        self.cam = None
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.num_cameras = self.cam_list.GetSize()
        print(f"Num detected = {self.num_cameras}")
        if self.num_cameras != 1:
            self.deinitialize_camera()
            print(f'Found {self.num_cameras} cameras where 1 was expected; aborting.')
            return

        self.cam = self.cam_list[0]
        print(f"self.cam = {self.cam}")
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()
        
    def deinitialize_camera(self):
        if self.cam is not None:
            self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def zern_to_dm(self, z, amp):
        assert type(z) == int, "first argument must be an integer (Zernike number)"
        assert type(amp) == float, "second argument must be a float (amplitude)"
        self.curr_dmc[:] = 0.0
        self.curr_dmc[z-1] = amp
        self.command_to_dm()

    def command_to_dm(self, verbose=True):
        """
        Send a command to the ShaneAO woofer.

        Parameters:
            amplitudes - list or np.ndarray
            The amplitude of each mode in [1, 2, ..., Nmodes], in order.
        """
        assert len(self.curr_dmc) == self.Nmodes, "wrong number of modes specified"
        assert np.all(np.abs(self.curr_dmc) <= 5.0), "sending out-of-bounds amplitudes"
        command = ",".join(map(str, self.curr_dmc))
        if verbose:
            print(f"DMC {command}.")
        warnings.warn("If you see this and you're at Lick, uncomment the lines defining and running shell_command.")
        # shell_command = ["ssh", "-Y", "gavel@shade.ucolick.org", "local/bin/imageSharpen", "-s", command]
        # subprocess.run(shell_command)

    def get_image(self, verbose=True):
        """
        Get an image off the lantern camera. 
        """
        configure_trigger(self.cam)
        img = acquire_image(self.cam, self.nodemap, self.nodemap_tldevice)
        reset_trigger(self.nodemap)
        return img

    def send_zeros(self, verbose=True):
        self.curr_dmc[:] = 0.0
        if verbose:
            print("Sending zeros.")
        self.command_to_dm(verbose=False)

    def try_intensities(self, img):
        if self.reader.save_intensities:
            return self.reader.get_intensities(img)
        else:
            return img

    def experiment(self, patterns):
        start_stamp = datetime_ms_now()
        self.send_zeros(verbose=False)
        img = self.get_image(verbose=False)
        time_stamps = []
        if self.reader.save_intensities:
            l = np.zeros((len(patterns), self.reader.nports))
        else:
            l = np.zeros((len(patterns), *self.reader.imgshape))
        for (i, p) in enumerate(tqdm(patterns)):
            time_stamps.append(time_ms_now())
            self.curr_dmc = p
            self.command_to_dm(verbose=False)
            img = self.get_image(verbose=False)
            l[i] = self.try_intensities(img)
        
        self.send_zeros(verbose=False)
        self.reader.save(f"dmc_{start_stamp}", patterns, verbose=False)
        self.reader.save(f"timestamps_{start_stamp}", np.array(time_stamps), verbose=False)
        self.reader.save(f"pl_{start_stamp}", l, verbose=False)

    # different types of experiment
    def sweep_mode(self, z, min_amp=-1.0, max_amp=1.0, step=0.1, prompt=False):
        amps = np.arange(min_amp, max_amp+step, step)
        patterns = np.zeros((len(amps), self.Nmodes))
        self.experiment(patterns)

    def sweep_all_modes(self, **kwargs):
        for z in range(1, self.Nmodes+1):
            print(f"Sweeping mode {z}")
            self.sweep_mode(z, **kwargs)

    def sweep_mode_combinations(self, amp=0.1, kmax=11, **kwargs):
        patterns = amp * np.array(
            reduce(lambda x, y: x + y, [
                [[float(i == j or i == j + k) for i in range(self.Nmodes)] for j in range(self.Nmodes - k)] for k in range(kmax)
            ])
        )
        self.experiment(patterns)

    def probe_signal(self, wait=0):
        probe = np.zeros((1, self.Nmodes))
        k = 0 # change this one to vary the mode of the probe signal
        probe[0,k] = 0.9 # change this one to vary the amplitude of the probe signal
        self.experiment(probe)

    def random_combinations(self, Niters, lim=1.0):
        inputzs = np.random.uniform(-lim, lim, (Niters, self.Nmodes))
        self.experiment(inputzs)

    def onsky(self, timeout):
        start_stamp = datetime_ms_now()
        self.probe_signal()
        lantern_images = []
        time_stamps = []
        tstart = time()
        while time() - tstart < timeout:
            try:
                time_stamps.append(time_ms_now())
                lantern_images.append(self.get_image(verbose=False))
                warnings.warn("If you see this and you're at Lick, delete the `sleep(0.1)` just below this.")
                sleep(0.1) # only for testing so I don't get ridiculous numbers of images
            except KeyboardInterrupt:
                print("Breaking out of loop")
                break

        print("Done collecting images")
        self.probe_signal()
        pl_to_save = []
        for img in tqdm(lantern_images):
            pl_to_save.append(self.try_intensities(img))
        self.reader.save(f"timestamps_{start_stamp}", time_stamps)
        self.reader.save(f"pl_{start_stamp}", np.array(pl_to_save))
                