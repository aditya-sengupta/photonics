import subprocess
import numpy as np
import dao
from functools import reduce
from time import sleep, time
from tqdm import tqdm, trange
import warnings
from matplotlib import pyplot as plt

from .utils import datetime_ms_now, time_ms_now, rms

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
	def __init__(self, reader, Nmodes=12):
		self.reader = reader
		self.Nmodes = Nmodes
		self.curr_dmc = np.zeros(Nmodes)
		self.im = dao.shm('/tmp/testShm.im.shm', np.zeros((520, 656)).astype(np.uint16))
		self.dit = dao.shm('/tmp/testShmDit.im.shm', np.zeros((1,1)).astype(np.float32))
		self.gain = dao.shm('/tmp/testShmGain.im.shm', np.zeros((1,1)).astype(np.float32))
		self.fps = dao.shm('/tmp/testShmFps.im.shm', np.zeros((1,1)).astype(np.float32))
  
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
		warnings.warn("If you see this and you're at Lick, uncomment the lines defining and running shell_command.")
		# shell_command = ["ssh", "-Y", "gavel@shade.ucolick.org", "local/bin/imageSharpen", "-s", command]
		# subprocess.run(shell_command)

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

	def try_intensities(self, img):
		if self.reader.save_intensities:
			return self.reader.get_intensities(img)
		else:
			return self.reader.crop_to_bounding_box(img)

	def experiment(self, patterns):
		start_stamp = datetime_ms_now()
		self.send_zeros(verbose=False)
		img = self.get_image()
		time_stamps = []
		if self.reader.save_intensities:
			pl_readout = np.zeros((len(patterns), self.reader.nports))
		else:
			bbox_shape = self.try_intensities(self.get_image()).shape
			pl_readout = np.zeros((len(patterns), *bbox_shape))
		for (i, p) in enumerate(tqdm(patterns)):
			time_stamps.append(time_ms_now())
			self.curr_dmc = p
			self.command_to_dm(verbose=False)
			img = self.get_image()
			pl_readout[i] = self.try_intensities(img)
		
		self.send_zeros(verbose=False)
		self.reader.save(f"dmc_{start_stamp}", patterns, verbose=False)
		self.reader.save(f"timestamps_{start_stamp}", np.array(time_stamps), verbose=False)
		self.reader.save(f"pl_{start_stamp}", pl_readout, verbose=False)

	def make_interaction_matrix(self, amp_calib=0.05):
		self.int_mat = np.zeros((self.reader.nports, self.Nmodes))
		for i in trange(self.Nmodes):
			self.zern_to_dm(i + 1, amp_calib, verbose=False)
			sleep(0.1)
			# don't go through try_intensities for this
			# i never want to do calibration on a full-frame lantern image
			s_push = normalize(self.reader.get_intensities(self.get_image()))
			self.zern_to_dm(i + 1, -amp_calib, verbose=False)
			sleep(0.1)
			s_pull = normalize(self.reader.get_intensities(self.get_image()))
			s = (s_push - s_pull) / (2 * amp_calib)
			self.int_mat[:,i] = s.ravel()
		
		self.send_zeros()
		self.compute_command_matrix()

	def compute_command_matrix(self, thres=1/30):
		self.cmd_mat = np.linalg.pinv(self.int_mat, thres)

	def linearity(self, modes_number=None, lim=1.0, step=0.1, plot=True):
		"""
		Make linearity curves for the WFS by sweeping (-lim, lim) in increments of `step`, and optionally plot them.

		Parameters
		----------
		modes_number : int
			The number of modes to check linearity for.
		lim : float, default
			The lower and upper limits on each mode, in DM units.
		step : float, default
			The increment used when sweeping through each mode, in DM units.
		plot : bool, defaullt
			Whether or not to plot the result.

		Returns
		-------
		amp_range : np.array [nAmps]
			The amplitudes used.
		responses : np.array [nModes, nAmps, nWFS]
			The full responses of the WFS to each mode.
		"""
		self.make_interaction_matrix()
		self.compute_command_matrix()
		if modes_number is None:
			nModes = self.Nmodes
		else:
			nModes = modes_number
		amp_range = np.arange(-lim, lim+2*np.finfo(float).eps, step)
		ref_image = self.get_image()
		ref_intensities = self.reader.get_intensities(ref_image)
		responses = np.zeros((nModes, len(amp_range), len(ref_intensities)))
		# interpretation: responses[i, j, k] contains the response in mode k to an input in mode i of amplitude amp_range[j].
		# The range on i is set by the number of modes the user requests.
		# The range on j is set by the limit and step size the user requests.
		# The range on k is inherent to the WFS.
		for i in range(1, nModes+1):
			for (j, amp) in enumerate(tqdm(amp_range)):
				self.zern_to_dm(i, float(amp), verbose=False)
				responses[i - 1, j, :] = self.reader.get_intensities(self.get_image())

		if plot:
			nModes = responses.shape[0]
			nrows = 3 
			# can set up more labels and so on, but this is good for a start.
			_, axs = plt.subplots(nrows, int(np.ceil(nModes / nrows)), sharex=True)
			for i in range(nModes):
				for k in range(responses.shape[2]):
					alpha = 1 if i == k else 0.1
					axs[k % nrows, k // nrows].plot(amp_range, responses[i,:,k], alpha=alpha)
					# axs[k % nrows, k // nrows].set_xlabel(i + 2)
			plt.show()
   
		return amp_range, responses

	def pseudo_cl_iteration(self, gain=0.1):
		# too experimental to put in loops and stuff!
		# if this ends up working, just loop it together with saving frames and stuff manually
		print(f"Initial error = {rms(self.curr_dmc)}")
		lantern_reading = self.cmd_mat @ self.reader.get_intensities(self.get_image())
		# does the sign of measured WF errors match the actual signs?
		sign_match = ''.join(map(lambda x: str(int(x)), (np.sign(lantern_reading * self.curr_dmc) * 2 + 2)/4))
		print(f"{sign_match}")
		self.curr_dmc -= gain * lantern_reading
		self.command_to_dm()
		print(f"Final error = {rms(self.curr_dmc)}")

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
		self.reader.save(f"timestamps_{start_stamp}", time_stamps)
		self.reader.save(f"pl_{start_stamp}", np.array(pl_to_save))
				