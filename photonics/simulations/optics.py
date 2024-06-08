"""
General optics setup that can be shared across WFS/AO system objects.
"""
import numpy as np
import hcipy as hc

class Optics:
	def __init__(self, lantern_fnumber=6.5, dm_basis="modal", num_pupil_px=60):
		self.dm_basis = dm_basis
		self.lantern_fnumber = lantern_fnumber
		self.mesh_extent = 512 # microns
		self.mesh_spacing = 1 # micron
		self.telescope_diameter = 1
		self.wl = 1.55e-6
		self.k = 2 * np.pi / self.wl
		self.pupil_grid = hc.make_pupil_grid(num_pupil_px, self.telescope_diameter)
		self.aperture = hc.evaluate_supersampled(hc.make_circular_aperture(self.telescope_diameter), self.pupil_grid, 6)
		spatial_resolution = self.wl * 1e-6 * self.lantern_fnumber # m/m = fraction
		q = spatial_resolution / self.mesh_spacing / 1e-6
		num_airy = self.mesh_extent * 1e-6 / (2 * spatial_resolution) # number of resolution elements
		num_px = 2 * q * num_airy
		if num_px % 2 == 0:
			num_airy *= (num_px + 1) / num_px
		self.focal_grid = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
		self.focal_propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=(self.lantern_fnumber * self.telescope_diameter))
		self.wf = hc.Wavefront(self.aperture, wavelength=self.wl)
		self.wf.total_power = 1
		self.im_ref = self.focal_propagator.forward(self.wf)
		self.norm = np.max(self.im_ref.intensity)
		self.turbulence_setup()
		self.dm_setup(dm_basis)
		self.zernike_basis = hc.mode_basis.make_zernike_basis(self.deformable_mirror.num_actuators, self.telescope_diameter, self.pupil_grid, starting_mode=2)
		
	def turbulence_setup(self, fried_parameter=0.1, outer_scale=50, velocity=10.0, seed=1):
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
 
	def zernike_to_phase(self, zernike, amplitude):
		amplitudes = np.zeros(self.deformable_mirror.num_actuators)
		zernike, amplitude = np.array(zernike), np.array(amplitude)
		amplitudes[zernike] = amplitude
		return self.zernike_basis.linear_combination(amplitudes)

	def phase_to_pupil(self, phase):
		aberration = np.exp(1j * phase)
		wavefront = hc.Wavefront(self.aperture * aberration, wavelength=self.wl)
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
