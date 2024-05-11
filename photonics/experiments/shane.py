import os
from os import path
import subprocess
import numpy as np
import dao
from functools import reduce
from time import sleep, time
from tqdm import tqdm, trange
import warnings
import paramiko

from .decometify import intensities_from_comet
from ..utils import date_now, datetime_ms_now, time_ms_now, rms, DATA_PATH

def normalize(x):
	return x / np.sum(x)

def save_telemetry(wait=0):
	warnings.warn("If you see this and you're at Lick, uncommand the line starting with subprocess.run")
	sleep(wait)
	"""subprocess.run
		"ssh", "-Y", 
		"user@shimmy.ucolick.org", "/opt/kroot/bin/modify",
		"-s", "saocon", "savedata=1"
	])
	"""

class ShaneLantern:
	def __init__(self, Nmodes=12):
		self.Nmodes = Nmodes
		self.Nports = 18
		self.curr_dmc = np.zeros(Nmodes)
		self.im = dao.shm('/tmp/testShm.im.shm', np.zeros((520, 656)).astype(np.uint16))
		self.dit = dao.shm('/tmp/testShmDit.im.shm', np.zeros((1,1)).astype(np.float32))
		self.gain = dao.shm('/tmp/testShmGain.im.shm', np.zeros((1,1)).astype(np.float32))
		self.fps = dao.shm('/tmp/testShmFps.im.shm', np.zeros((1,1)).astype(np.float32))
		# self.setup_paramiko()
		subdir = f"pl_{date_now()}"
		self.subdir = subdir
		if not os.path.isdir(self.directory):
			os.mkdir(self.directory)
		print(f"Path for data saving set to {self.directory}")

	@property
	def directory(self):
		return path.join(DATA_PATH, self.subdir)

	def setup_subprocess(self):
		self.subproc = subprocess.Popen(["ssh", "-Y", "user@karnak.ucolick.org"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
 
	def setup_paramiko(self):
		host = "karnak.ucolick.org"
		username = "user"
		password = "yam != spud"
  
		self.client = paramiko.client.SSHClient()
		self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.client.connect(host, 22, username=username, password=password)
		self.channel = self.client.get_transport().open_session()
		self.channel.get_pty()
		self.channel.invoke_shell()
	
	def filepath(self, fname, ext=None):
		"""
		The path we want to save/load data from or to.
		"""
		if ext is None:
			ext = self.ext
		return path.join(DATA_PATH, self.subdir, fname + "." + ext)
  
	def measure_dark(self):
		darks = []
		for _ in trange(10):
			darks.append(self.im.get_data(check=True).astype(float))
			sleep(self.dit.get_data()[0][0] / 1e6)

		self.dark = np.mean(darks, axis=0)

	def get_exp(self):
		return self.dit.get_data()

	def set_exp(self, texp):
		self.dit.set_data(self.dit.get_data() * 0 + texp)
  
	def get_gain(self):
		return self.gain.get_data()

	def set_gain(self, gain):
		self.gain.set_data(self.dit.get_data() * 0 + gain)
  
	def get_fps(self):
		return self.fps.get_data()

	def set_fps(self, fps):
		self.fps.set_data(self.fps.get_data() * 0 + fps)

	def zern_to_dm(self, z, amp, verbose=True):
		assert isinstance(z, int), "first argument must be an integer (Zernike number)"
		assert isinstance(amp, float), "second argument must be a float (amplitude)"
		self.curr_dmc[:] = 0.0
		self.curr_dmc[z-1] = amp
		self.command_to_dm(verbose=verbose)
  
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
		# warnings.warn("If you see this and you're at Lick, uncomment the lines defining and running shell_command.")
		shell_command = ["ssh", "-Y", "user@karnak.ucolick.org", "/home/user/ShaneAO/shade/imageSharpen", "-s", command]
		subprocess.run(shell_command)
	
	def get_image(self):
		"""
		Get an image off the lantern camera. 
		"""
		sleep(self.dit.get_data()[0][0] / 1e6)
		return self.im.get_data(check=True).astype(float) - self.dark

	def send_zeros(self, verbose=True):
		self.curr_dmc[:] = 0.0
		if verbose:
			print("Sending zeros.")
		self.command_to_dm(verbose=False)

	def experiment(self, patterns):
		start_stamp = datetime_ms_now()
		self.send_zeros(verbose=False)
		img = self.get_image()
		time_stamps = []
		bbox_shape = self.get_image().shape
		pl_readout = np.zeros((len(patterns), *bbox_shape))
		for (i, p) in enumerate(tqdm(patterns)):
			time_stamps.append(time_ms_now())
			self.curr_dmc = p
			self.command_to_dm(verbose=False)
			img = self.get_image()
			pl_readout[i] = img
		
		self.send_zeros(verbose=False)
		self.save(f"dmc_{start_stamp}", patterns, verbose=False)
		self.save(f"timestamps_{start_stamp}", np.array(time_stamps), verbose=False)
		self.save(f"pl_{start_stamp}", pl_readout, verbose=False)
		return np.array(list(map(intensities_from_comet, pl_readout)))

	def make_interaction_matrix(self, amp_calib=0.05, nm=None):
		if nm is None:
			nm = self.Nmodes
		self.int_mat = np.zeros((self.Nports, nm))
		pushes = []
		pulls = []
		for i in trange(nm):
			self.zern_to_dm(i + 1, amp_calib, verbose=False)
			sleep(0.1)
			s_push = intensities_from_comet(self.get_image())
			pushes.append(s_push)
			self.zern_to_dm(i + 1, -amp_calib, verbose=False)
			sleep(0.1)
			s_pull = intensities_from_comet(self.get_image())
			pulls.append(s_pull)
			s = (s_push - s_pull) / (2 * amp_calib)
			self.int_mat[:,i] = s.ravel()
		
		self.save(f"pushes_amp_{amp_calib}_datetime_{datetime_ms_now()}", np.array(pushes))
		self.save(f"pulls_amp_{amp_calib}_datetime_{datetime_ms_now()}", np.array(pulls))
		self.save(f"intmat_amp_{amp_calib}_datetime_{datetime_ms_now()}", self.int_mat)
		self.send_zeros()
		self.compute_command_matrix(amp_calib)

	def compute_command_matrix(self, amp_calib, thres=1/30):
		self.cmd_mat = np.linalg.pinv(self.int_mat, thres)
		self.save(f"cmdmat_amp_{amp_calib}_datetime_{datetime_ms_now()}", self.cmd_mat)

	def pseudo_cl_iteration(self, gain=0.1):
		# too experimental to put in loops and stuff!
		# if this ends up working, just loop it together with saving frames and stuff manually
		print(f"Initial error = {rms(self.curr_dmc)}")
		lantern_reading = np.zeros(self.Nmodes)
		lr = self.cmd_mat @ intensities_from_comet(self.get_image())
		lantern_reading[:len(lr)] = lr
		# does the sign of measured WF errors match the actual signs?
		sign_match = ''.join(map(lambda x: str(int(x)), (np.sign(lantern_reading * self.curr_dmc) * 2 + 2)/4))
		print(f"{sign_match}")
		self.curr_dmc -= gain * lantern_reading
		self.command_to_dm()
		print(f"Final error = {rms(self.curr_dmc)}")

	# different types of experiment
	def sweep_mode(self, z, min_amp=-1.0, max_amp=1.0, step=0.1):
		amps = np.arange(min_amp, max_amp+2*step, step)
		patterns = np.zeros((len(amps), self.Nmodes))
		patterns[:,z-1] = amps
		return amps, self.experiment(patterns)

	def sweep_all_modes(self, min_amp=-1.0, max_amp=1.0, step=0.1):
		amps = np.arange(min_amp, max_amp+step, step)
		patterns = np.zeros((len(amps) * self.Nmodes + 1, self.Nmodes))
		for z in range(1, self.Nmodes+1):
			patterns[len(amps)*(z-1):len(amps)*(z),z-1] = amps
		return amps, self.experiment(patterns)

	def sweep_mode_combinations(self, amp=0.1, kmax=11, **kwargs):
		patterns = amp * np.array(
			reduce(lambda x, y: x + y, [
				[[float(i == j or i == j + k) for i in range(self.Nmodes)] for j in range(self.Nmodes - k)] for k in range(kmax)
			])
		)
		return patterns, self.experiment(patterns)

	def probe_signal(self, wait=0):
		probe = np.zeros((1, self.Nmodes))
		k = 0 # change this one to vary the mode of the probe signal
		probe[0,k] = 0.9 # change this one to vary the amplitude of the probe signal
		self.experiment(probe)

	def random_combinations(self, Niters, lim=1.0):
		inputzs = np.random.uniform(-lim, lim, (Niters+1, self.Nmodes))
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
				lantern_images.append(self.get_image())
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
		self.save(f"timestamps_{start_stamp}", time_stamps)
		self.save(f"pl_{start_stamp}", np.array(pl_to_save))
				
	def save(self, fname, data, ext="npy", verbose=True):
		fpath = self.filepath(fname, ext=ext)
		if verbose:
			print(f"Saving data to {fpath}")
		np.save(self.filepath(fname, ext=ext), data)