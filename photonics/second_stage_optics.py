import numpy as np
import hcipy as hc
from copy import copy
from tqdm import trange
from matplotlib import pyplot as plt
from .utils import rms
from .pyramid_optics import PyramidOptics
from .lantern_optics import LanternOptics

class SecondStageOptics:
	def __init__(self, lantern_fnumber=6.5):
		self.optics_setup(lantern_fnumber)
		self.turbulence_setup()
		self.dm_setup()
		self.pyramid_optics = PyramidOptics(self)
		self.lantern_optics = LanternOptics(self)
		
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
		
	def turbulence_setup(self, fried_parameter=0.1, outer_scale=50, velocity=10.):
		Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2
		self.layer = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)
		
	def dm_setup(self):
		#make the DM
		num_actuators = 9
		actuator_spacing = self.telescope_diameter / num_actuators
		influence_functions = hc.make_gaussian_influence_functions(self.pupil_grid, num_actuators, actuator_spacing)
		self.deformable_mirror = hc.DeformableMirror(influence_functions)
		#modes = hc.make_zernike_basis(num_actuators ** 2, self.telescope_diameter, self.pupil_grid, starting_mode=2)
		#self.deformable_mirror = hc.DeformableMirror(modes)
	
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
			
	def zernike_to_focal(self, zernike, amplitude):
		return self.prop(self.phase_to_pupil(self.zernike_to_phase(zernike, amplitude)))

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

	def pyramid_correction(self, num_iterations=200, dt=1./800, gain = 0.1, leakage = 0.999, plot=False):
		"""
  		Soon this will be the general CL test and we'll be able to turn on one WFS or the other
    	"""
		correction_results = {
			"wavefront_after_dm_errors" : [],
			"wavefronts_after_dm" : [],
			"pyramid_readings" : [],
			"dm_shapes" : [],
			"point_spread_functions" : [],
			"strehl_ratios" : [],
			"phases_for" : []
		}
		self.layer.reset()
		self.layer.t = 0
		for timestep in trange(num_iterations):
			wf_after_dm = self.wavefront_after_dm(timestep * dt)
			wfs_image = self.pyramid_optics.readout(wf_after_dm)
			diff_image = wfs_image - self.pyramid_optics.image_ref
			pyramid_reading = self.pyramid_optics.command_matrix.dot(diff_image)
			self.deformable_mirror.actuators = leakage*self.deformable_mirror.actuators - gain*pyramid_reading
			wf_focal = self.focal_propagator.forward(wf_after_dm)
			strehl_foc = hc.get_strehl_from_focal(wf_focal.intensity/self.norm,self.im_ref.intensity/self.norm)

			correction_results["wavefront_after_dm_errors"].append(float(rms(wf_after_dm.phase * self.aperture)))
			correction_results["wavefronts_after_dm"].append(wf_after_dm.copy())
			correction_results["pyramid_readings"].append(pyramid_reading)
			correction_results["dm_shapes"].append(copy(self.deformable_mirror.surface))
			correction_results["point_spread_functions"].append(wf_focal.copy())
			correction_results["strehl_ratios"].append(float(strehl_foc))
			correction_results["phases_for"].append(self.layer.phase_for(self.wl))
		
		#plot the results
		if plot:
			fig = plt.figure(figsize=(15,8))

			plt.subplot(1,3,1)
			plt.title(r'DM surface shape ($\mathrm{\mu}$m)')
			hc.imshow_field(self.aperture*self.deformable_mirror.surface/(1e-6), vmin=-1, vmax=1, cmap='bwr')
			plt.colorbar(fraction=0.046, pad=0.04)

			plt.subplot(1,3,2)
			plt.title('Residual wavefront error (rad)')
			res = wf_after_dm.phase*self.aperture
			hc.imshow_field(res, cmap='RdBu')
			plt.colorbar(fraction=0.046, pad=0.04)

			plt.subplot(1,3,3)
			plt.title('Focal Plane Image (Strehl = %.2f)'% (np.mean(np.asarray(correction_results["strehl_ratios"]))))
			hc.imshow_field(np.log10(wf_focal.intensity/self.norm), cmap='inferno')
			plt.colorbar(fraction=0.046, pad=0.04)
			plt.tight_layout()
			plt.show()
		return correction_results # {k : np.array(correction_results[k]) for k in correction_results}


	