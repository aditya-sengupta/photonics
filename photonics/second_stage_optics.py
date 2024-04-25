import numpy as np
import hcipy as hc
from tqdm import trange
from matplotlib import pyplot as plt
from .pyramid_optics import PyramidOptics

class SecondStageOptics:
	def __init__(self, lo):
		self.optics_setup_lantern(lo)
		self.turbulence_setup()
		self.dm_setup()
		self.pyramid = PyramidOptics(self)
		
	def optics_setup_lantern(self, lo):
		self.telescope_diameter = 1
		self.wl = lo.wl * 1e-6
		self.k = 2 * np.pi / self.wl
		self.f_number = lo.f_number
		self.pupil_grid = hc.make_pupil_grid(60, self.telescope_diameter)
		self.aperture = hc.evaluate_supersampled(hc.make_circular_aperture(self.telescope_diameter), self.pupil_grid, 6)
		spatial_resolution = self.wl * 1e-6 * self.f_number # m/m = fraction
		q = spatial_resolution / lo.mesh.ds / 1e-6
		num_airy = lo.mesh.xw * 1e-6 / (2 * spatial_resolution) # number of resolution elements
		num_px = 2 * q * num_airy
		if num_px % 2 == 0:
			num_airy *= (num_px + 1) / num_px
		self.focal_grid = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
		self.focal_propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=(self.f_number * self.telescope_diameter))
		self.pupil_wf_ref = lo.zernike_to_pupil(4, 0.0)
		self.focal_wf_ref = self.focal_propagator(self.pupil_wf_ref)
		self.wf = hc.Wavefront(self.aperture, wavelength=self.wl)
		self.wf.total_power = 1
		self.im_ref = self.focal_propagator.forward(self.wf)
		self.norm = np.max(self.im_ref.intensity)
		
	def optics_setup(self):
		self.telescope_diameter = 1  
		self.pupil_grid = hc.make_pupil_grid(60, diameter=self.telescope_diameter)
		self.aperture = hc.evaluate_supersampled(hc.make_circular_aperture(self.telescope_diameter), self.pupil_grid, 6)
		self.wl = 1.55e-6
		self.k = 2*np.pi/self.wl
		self.focal_grid = hc.make_focal_grid(q=4, num_airy=20,spatial_resolution=self.wl/self.telescope_diameter)
		self.focal_propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)
		self.wf = hc.Wavefront(self.aperture, wavelength=self.wl)
		self.wf.total_power = 1
		self.im_ref = self.focal_propagator.forward(self.wf)
		self.norm = np.max(self.im_ref.intensity)
		
	def turbulence_setup(self):
		fried_parameter = 0.13 # meters
		outer_scale = 50 #  meters
		velocity = 10. # wind speed in m/s.
		Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2

		# make our atmospheric turbulence layer
		self.layer = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)
		
	def dm_setup(self):
		#make the DM
		num_actuators = 9
		actuator_spacing = self.telescope_diameter / num_actuators
		influence_functions = hc.make_gaussian_influence_functions(self.pupil_grid, num_actuators, actuator_spacing)
		self.deformable_mirror = hc.DeformableMirror(influence_functions)
		
	def pyramid_correction(self, num_iterations=40, dt=1./800, gain = 0.6, leakage = 0.999):
		sr=[]

		self.layer.reset()
		self.layer.t = 0
		for timestep in trange(num_iterations):
			#get a clean wavefront
			wf_in = self.wf.copy()

			#evolve the atmospheric turbulence
			self.layer.t = timestep*dt

			#pass the wavefront through the turbulence
			wf_after_atmos = self.layer.forward(wf_in)

			#pass the wavefront through the DM for correction
			wf_after_dm = self.deformable_mirror.forward(wf_after_atmos)

			#send the wavefront containing the residual wavefront error to the PyWFS and get slopes
			wfs_image = self.pyramid.readout(wf_after_dm)
			diff_image = wfs_image - self.pyramid.image_ref

			#Leaky integrator to calculate new DM commands
			self.deformable_mirror.actuators =  leakage*self.deformable_mirror.actuators - gain * self.pyramid.command_matrix.dot(diff_image)

			# Propagate to focal plane
			wf_focal = self.focal_propagator.forward(wf_after_dm)

			#calculate the strehl ratio to use as a metric for how well the AO system is performing.
			strehl_foc = hc.get_strehl_from_focal(wf_focal.intensity/self.norm,self.im_ref.intensity/self.norm)
			sr.append(strehl_foc)
			
		#plot the results
		fig = plt.figure(figsize=(15,8))

		plt.subplot(1,3,1)
		plt.title(r'DM surface shape ($\mathrm{\mu}$m)')
		hc.imshow_field(self.aperture*self.deformable_mirror.surface/(1e-6), vmin=-1, vmax=1, cmap='bwr')
		plt.colorbar(fraction=0.046, pad=0.04)

		plt.subplot(1,3,2)
		plt.title('Residual wavefront error (rad)')
		res=wf_after_dm.phase*self.aperture
		hc.imshow_field(res, cmap='RdBu')
		plt.colorbar(fraction=0.046, pad=0.04)

		plt.subplot(1,3,3)
		plt.title('Focal Plane Image (Strehl = %.2f)'% (np.mean(np.asarray(sr))))
		hc.imshow_field(np.log10(wf_focal.intensity/self.norm), cmap='inferno')
		plt.colorbar(fraction=0.046, pad=0.04)
		plt.tight_layout()
		plt.show()
		return list(map(float, sr))
	