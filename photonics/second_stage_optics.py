import numpy as np
import hcipy as hc
import sys
from copy import copy
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from .utils import rms
from .pyramid_optics import PyramidOptics
from .lantern_optics import LanternOptics
from .wfs_filter import HighPassFilter, LowPassFilter

class SecondStageOptics:
	def __init__(self, lantern_fnumber=6.5, n_filter=9, f_cutoff=30, f_loop=100, dm_basis="zonal", ncpa_z=4, ncpa_a=0.0):
		a = np.exp(-2 * np.pi * f_cutoff / f_loop)
		self.pyramid_filter = HighPassFilter(n_filter, a)
		self.lantern_filter = LowPassFilter(n_filter, a)
		self.f_loop = f_loop
		self.dt = 1 / f_loop
		self.optics_setup(lantern_fnumber)
		self.turbulence_setup()
		self.dm_setup(dm_basis)
		self.pyramid_optics = PyramidOptics(self, dm_basis)
		self.lantern_optics = LanternOptics(self)
		self.zernike_basis = hc.mode_basis.make_zernike_basis(n_filter, self.telescope_diameter, self.pupil_grid, starting_mode=2)
		self.ncpa_z, self.ncpa_a = ncpa_z, ncpa_a
		self.ncpa = hc.Wavefront(np.exp(1j * self.zernike_basis[ncpa_z] * ncpa_a) * self.aperture, wavelength=self.wl)
		
	def optics_setup(self, lantern_fnumber=6.5):
		self.lantern_fnumber = lantern_fnumber
		self.mesh_extent = 512 # microns
		self.mesh_spacing = 1 # micron
		self.telescope_diameter = 1
		self.wl = 1.55e-6
		self.k = 2 * np.pi / self.wl
		self.pupil_grid = hc.make_pupil_grid(60, self.telescope_diameter)
		self.aperture = hc.evaluate_supersampled(hc.make_circular_aperture(self.telescope_diameter), self.pupil_grid, 6)
		spatial_resolution = self.wl * 1e-6 * self.lantern_fnumber # m/m = fraction
		q = spatial_resolution / self.mesh_spacing / 1e-6
		num_airy = self.mesh_extent * 1e-6 / (2 * spatial_resolution) # number of resolution elements
		num_px = 2 * q * num_airy
		if num_px % 2 == 0:
			num_airy *= (num_px + 1) / num_px
		self.focal_grid = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
		self.focal_propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=(self.lantern_fnumber * self.telescope_diameter))
		self.pupil_wf_ref = self.zernike_to_pupil(4, 0.0)
		self.focal_wf_ref = self.focal_propagator(self.pupil_wf_ref)
		self.wf = hc.Wavefront(self.aperture, wavelength=self.wl)
		self.wf.total_power = 1
		self.im_ref = self.focal_propagator.forward(self.wf)
		self.norm = np.max(self.im_ref.intensity)
		
	def turbulence_setup(self, fried_parameter=0.5, outer_scale=50, velocity=10.0, seed=1):
		Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter)
		self.layer = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity, seed=seed)
		
	def dm_setup(self, dm_basis, num_actuators=9):
		if dm_basis == "zonal":
			actuator_spacing = self.telescope_diameter / num_actuators
			influence_functions = hc.make_gaussian_influence_functions(self.pupil_grid, num_actuators, actuator_spacing)
			self.deformable_mirror = hc.DeformableMirror(influence_functions)
		elif dm_basis == "modal":
			modes = hc.make_zernike_basis(num_actuators ** 2, self.telescope_diameter, self.pupil_grid, starting_mode=2)
			self.deformable_mirror = hc.DeformableMirror(modes)
		else:
			raise NameError("DM basis needs to be zonal or modal")
 
	# awful design. Need a refactor so these don't also live in lanternoptics, after I've got GS working.
	def zernike_to_phase(self, zernike, amplitude):
		if isinstance(zernike, list) and isinstance(amplitude, list):
			phase = hc.Field(sum(a * hc.mode_basis.zernike_ansi(z, D=self.telescope_diameter)(self.pupil_grid).shaped for (z, a) in zip(zernike, amplitude)).ravel(), self.pupil_grid)
		else:
			phase = amplitude * hc.mode_basis.zernike_ansi(zernike, D=self.telescope_diameter)(self.pupil_grid)
		return phase

	def phase_to_pupil(self, phase):
		aberration = np.exp(1j * phase)
		wavefront = hc.Wavefront(self.aperture * aberration, wavelength=self.wl*1e-6)
		return wavefront
	
	def zernike_to_pupil(self, zernike, amplitude):
		return self.phase_to_pupil(self.zernike_to_phase(zernike, amplitude))
			
	def wavefront_after_dm(self, t):
		"""
		Generates the wavefront after the DM at a time "t".
		This uses the current shape of self.deformable_mirror
		so just repeatedly calling it in open loop won't create cleaner wavefronts.
  		"""
		wf_in = self.wf.copy()
		self.layer.t = t
		wf_after_atmos = self.layer.forward(wf_in)
		return self.deformable_mirror.forward(wf_after_atmos)

	def correction(self, num_iterations=200, gain=0.1, leakage=0.999, plot=False, two_stage_niter=100):
		"""
		Simulates a full AO loop.
    	"""
		correction_results = {
			"wavefronts_after_dm" : [],
			"pyramid_readings" : [],
			"dm_shapes" : [],
			"point_spread_functions" : [],
			"strehl_ratios" : [],
			"lantern_readings" : [],
   			"lantern_zernikes_truth" : [],
			"lantern_zernikes_measured" : []
		}
		self.layer.reset()
		self.layer.t = 0
		with tqdm(range(num_iterations), file=sys.stdout) as progress:
			for timestep in progress:
				wf_after_dm = self.wavefront_after_dm(timestep * self.dt)
				pyramid_reading = self.pyramid_optics.reconstruct(wf_after_dm)
				dm_command = np.copy(pyramid_reading)
				if timestep > two_stage_niter:
					hpf_reading = self.pyramid_filter(pyramid_reading[:self.pyramid_filter.n])
					dm_command[:self.pyramid_filter.n] = hpf_reading
				if self.ncpa_a != 0:
					wf_focal = self.focal_propagator.forward(
						hc.Wavefront(
							wf_after_dm.electric_field * self.ncpa.electric_field,
							wavelength = self.wl
						)
					)
				else:
					wf_focal = self.focal_propagator.forward(wf_after_dm)

				if timestep == two_stage_niter:
					tqdm.write(f"Closing second-stage loop at iteration {timestep}")
				if timestep > two_stage_niter:
					lantern_zernikes_truth = self.zernike_basis.coefficients_for(wf_after_dm.phase)
					lantern_zernikes_truth[self.ncpa_z] += self.ncpa_a
					correction_results["lantern_zernikes_truth"].append(lantern_zernikes_truth)
					lantern_reading = np.abs(self.lantern_optics.lantern_coeffs(wf_focal)) ** 2
					lantern_zernikes_measured = self.lantern_optics.command_matrix @ lantern_reading
					lpf_reading = self.lantern_filter(lantern_zernikes_measured)
					correction_results["lantern_zernikes_measured"].append(lantern_zernikes_measured)
					dm_command[:self.lantern_filter.n] += lpf_reading

				self.deformable_mirror.actuators = leakage * self.deformable_mirror.actuators - gain * dm_command
				correction_results["wavefronts_after_dm"].append(wf_after_dm.copy())
				correction_results["pyramid_readings"].append(pyramid_reading)
				correction_results["dm_shapes"].append(copy(self.deformable_mirror.surface))
				correction_results["point_spread_functions"].append(wf_focal.copy())
				strehl_foc = hc.get_strehl_from_focal(wf_focal.intensity/self.norm,self.im_ref.intensity/self.norm)
				correction_results["strehl_ratios"].append(float(strehl_foc))
				progress.set_postfix(strehl=f"{float(strehl_foc):.3f}")
  
		return correction_results
