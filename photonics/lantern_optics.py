import numpy as np
import hcipy as hc
import lightbeam as lb
from matplotlib import pyplot as plt
from tqdm import trange
from .utils import PROJECT_ROOT

class LanternOptics:
	def __init__(self, f_number=10):
		# coupling = unaberrated PSF radius / input core radius
		self.wl = 1.55
		self.telescope_diameter = 10.0
		self.final_scale = 8 # tapering factor of lantern
		self.cladding_radius = 37/2 * self.final_scale
		self.core_offset = self.cladding_radius / 2.5 # offset of cores from origin
		self.nclad = 1.4504
		self.ncore = self.nclad + 0.01036 # lantern core refractive index
		self.njack = np.sqrt(self.nclad**2-0.125**2) # jacket index
		self.rcore = 2.2
		self.z_ex = 60_000
		self.lant = lb.optics.make_lant19(self.core_offset,self.rcore,self.cladding_radius,0,self.z_ex, (self.ncore,self.nclad,self.njack),final_scale=1/self.final_scale)
		self.make_mesh(self.lant)
		self.setup_hcipy(f_number)
		self.launch_fields = [
			lb.normalize(lb.lpfield(self.xg-pos[0], self.yg-pos[1], 0, 1, self.rcore, self.wl, self.ncore, self.nclad))
			for pos in self.lant.init_core_locs
		]
		self.plotting_launch_fields = [
			lb.normalize(lb.lpfield(self.xg-pos[0], self.yg-pos[1], 0, 1, 5 * self.rcore, self.wl, self.ncore, self.nclad))
			for pos in self.lant.init_core_locs
		]
		self.lbprop = lb.Prop3D(self.wl, self.mesh, self.lant, self.nclad)
		self.lantern_basis = np.array([lf.ravel() for lf in self.launch_fields]).T
		self.lantern_reverse = np.linalg.inv(self.lantern_basis.T @ self.lantern_basis) @ self.lantern_basis.T
		out = np.zeros_like(self.xg)
		self.lant.set_IORsq(out, self.z_ex)
		out = out[self.mesh.PML:-self.mesh.PML,self.mesh.PML:-self.mesh.PML]
		self.input_footprint = np.where(out >= self.nclad ** 2)
		self.extent_x = (np.min(self.input_footprint[0]), np.max(self.input_footprint[0]))
		self.extent_y = (np.min(self.input_footprint[1]), np.max(self.input_footprint[1]))
		
	def setup_hcipy(self, f_number):
		self.f_number = f_number
		self.pupil_grid = hc.make_pupil_grid(60, self.telescope_diameter)
		self.aperture = hc.evaluate_supersampled(hc.make_circular_aperture(self.telescope_diameter), self.pupil_grid, 6)
		spatial_resolution = self.wl * 1e-6 * f_number # m/m = fraction
		q = spatial_resolution / self.mesh.ds / 1e-6
		num_airy = self.mesh.xw * 1e-6 / (2 * spatial_resolution) # number of resolution elements
		num_px = 2 * q * num_airy
		if num_px % 2 == 0:
			num_airy *= (num_px + 1) / num_px
		self.focal_grid = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=spatial_resolution)
		self.prop = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=(f_number * self.telescope_diameter))
		self.pupil_wf_ref = self.zernike_to_pupil(4, 0.0)
		self.focal_wf_ref = self.prop(self.pupil_wf_ref)
		
	def input_to_2d(self, input_efield, zoomed=True):
		"""
		Takes in an input electric field as a 1D array (coordinates represented by input_footprint) and fills it in to a 2D grid for plotting.
		
		Can either be "zoomed" into the footprint (for plotting), or not (for backpropagating through HCIPy optics).
		"""
		if zoomed:
			xl, yl = self.extent_x[1] - self.extent_x[0] + 1, self.extent_y[1] - self.extent_y[0] + 1
			xm, ym = self.extent_x[0], self.extent_y[0]
		else:
			xl, yl = self.focal_grid.shape
			xm, ym = 0, 0
			
		input_efield_2d = np.zeros((xl, yl), dtype=np.complex64)
		input_efield_2d[self.input_footprint[0] - xm, self.input_footprint[1] - ym] = input_efield
		return input_efield_2d
		

	def make_mesh(self, lant):
		mesh = lb.RectMesh3D(
			xw = 512, # um
			yw = 512, # um
			zw = self.z_ex, # um
			ds = 1, # um
			dz = 50, # um
			PML = 8 # grid units
		)
		lant.set_sampling(mesh.xy)
		xg, yg = mesh.grids_without_pml()
		self.mesh = mesh
		self.w = self.mesh.xy.get_weights()[mesh.PML:-mesh.PML,mesh.PML:-mesh.PML]
		self.xg = xg
		self.yg = yg
		
	def sanitize_output(self, x):
		x = x[self.input_footprint]
		return x / np.linalg.norm(x)
		
	def propagate_backwards(self, doplot=True):
		outputs = []
		for (i, lf) in enumerate(self.launch_fields):
			print(f"Illuminating core {i}")
			if doplot:
				plt.imshow(np.abs(lf) ** 2)
				plt.show()
			u = self.lbprop.prop2end(lf)[self.mesh.PML:-self.mesh.PML,self.mesh.PML:-self.mesh.PML]
			outputs.append(u)
			if doplot:
				output_intensity = np.abs(self.input_to_2d(u[self.input_footprint])) ** 2
				plt.imshow(output_intensity)
				plt.show()
		
		np.save(PROJECT_ROOT + "/data/backprop_19_{:.2f}.npy".format(self.f_number), np.array(outputs))
		
	def load_outputs(self):
		outputs = np.load(PROJECT_ROOT + "/data/backprop_19_{:.2f}.npy".format(self.f_number))
		self.outputs = np.array([self.sanitize_output(x) for x in outputs])
		self.projector = np.linalg.inv(self.outputs @ self.outputs.T) @ self.outputs

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

	def lantern_output(self, focal_field):
		profile_to_project = focal_field.electric_field.shaped[self.input_footprint]
		coeffs = self.projector @ profile_to_project
		projected = self.input_to_2d(coeffs @ self.outputs)
		lantern_reading = sum(c * lf for (c, lf) in zip(coeffs, self.launch_fields))
		return coeffs, projected, lantern_reading

	def show_lantern_output(self, zernike, amplitude):
		if not isinstance(zernike, list):
			zernike = [zernike]
			amplitude = [amplitude]
		phase = hc.Field(sum(a * hc.mode_basis.zernike_ansi(z, D=self.telescope_diameter)(self.pupil_grid).shaped for (z, a) in zip(zernike, amplitude)).ravel(), self.pupil_grid)
		focal_field = self.prop(self.phase_to_pupil(phase))
		coeffs, projected, lantern_reading = self.lantern_output(focal_field)
		fig, axs = plt.subplots(1, 3)
		fig.suptitle(f"Lantern response for Zernike {zernike}, amplitude {amplitude}")
		fig.subplots_adjust(top=1.4, bottom=0.0)
		for ax in axs:
			ax.set_xticks([])
			ax.set_yticks([])
		axs[0].imshow(phase.shaped)
		axs[0].set_title("Phase screen")
		axs[1].imshow(np.abs(focal_field.intensity.shaped))
		axs[1].set_title("Lantern input")
		axs[2].imshow(np.abs(lantern_reading))
		axs[2].set_title("Lantern output")
		plt.show()
		
	def lantern_reading(self, zernike, amplitude):
		coeffs, _, _ = self.lantern_output(self.zernike_to_focal(zernike, amplitude))
		return np.abs(coeffs) ** 2
	
	def make_intcmd(self, nzern=6):
		poke_amplitude = 1e-10
		pokes = []
		for i in trange(1, nzern+1):
			amp_plus = self.lantern_reading(i, poke_amplitude)
			amp_minus = self.lantern_reading(i, -poke_amplitude)
			pokes.append((amp_plus - amp_minus) / (2 * poke_amplitude))

		self.int_matrix = np.array(pokes).T
		self.cmd_matrix = np.linalg.pinv(self.int_matrix, rcond=1e-5)
		self.flat_amp = self.lantern_reading(1, 0.0)
		
	def make_linearity(self, nzern=6, lim=0.1, step=None):
		if step is None:
			step = lim / 4
		amplitudes = np.arange(-lim, lim*1.001, step)
		linearity_responses = np.zeros((nzern, len(amplitudes), nzern))
		for z in trange(1, nzern+1):
			for (j,a) in enumerate(amplitudes):
				flattened = self.lantern_reading(z,a) - self.flat_amp
				response = self.cmd_matrix @ flattened
				linearity_responses[z-1,j,:nzern] = response
		
		return amplitudes, linearity_responses

	def show_linearity(self, amplitudes, linearity_responses):
		nzern = linearity_responses.shape[0]
		fig, axs = plt.subplots(int(np.ceil(nzern // 3)), 3, sharex=True, sharey=True, figsize=(9, 9))
		plt.suptitle(f"Photonic lantern linearity curves (rad), f/{self.f_number}")
		for i in range(nzern):
			r, c = i // 3, i % 3
			axs[r][c].set_ylim([min(amplitudes), max(amplitudes)])
			axs[r][c].title.set_text(f"Z{i+1}")
			for j in range(nzern):
				alpha = 1 if i == j else 0.1
				axs[r][c].plot(amplitudes, linearity_responses[i,:,j], alpha=alpha)
		plt.show()
		
	def forward(self, pupil):
		focal = self.prop.forward(pupil)
		_, _, reading = self.lantern_output(focal)
		return hc.Wavefront(hc.Field(reading.ravel(), self.focal_grid), wavelength=self.wl)
	
	def backward(self, reading):
		coeffs = self.lantern_reverse @ reading.electric_field
		lantern_input = self.input_to_2d(coeffs @ self.outputs, zoomed=False)
		focal_wf = hc.Wavefront(hc.Field(lantern_input.ravel(), self.focal_grid), wavelength=self.wl)
		pupil_field = self.prop.backward(focal_wf)
		return pupil_field
		 
	def GS(self, img, niter=10):
		"""
		Gerchberg-Saxton algorithm
		"""        
		# Normalization
		img /= img.sum()
		
		# Amplitude in - measured (or assumed to be known)
		measuredAmplitude_in = self.pupil_wf_ref
		# Amplitude out - measured
		measuredAmplitude_out = np.sqrt(img) 
		
		# Forward Propagation in WFS assuming 0 phase
		EM_out = self.forward(measuredAmplitude_in)
		# replacing by known amplitude
		phase_out_0 = EM_out.phase
		EM_out = measuredAmplitude_out * np.exp(1j*phase_out_0)
		# Back Propagation in PWFS
		EM_in = self.backward(hc.Wavefront(EM_out, wavelength=self.wl))
		for k in trange(niter):
			# replacing by known amplitude
			phase_in_k = EM_in.phase
			EM_in = hc.Wavefront(measuredAmplitude_in.electric_field*np.exp(1j*phase_in_k),wavelength=self.wl)
			# Lantern forward propagation
			EM_out = self.forward(EM_in)
			# replacing by known amplitude
			phase_out_k = EM_out.phase
			EM_out = hc.Wavefront(measuredAmplitude_out*np.exp(1j*phase_out_k), wavelength=self.wl)
			# Lantern backward propagation    
			EM_in = self.backward(EM_out)
			
		return EM_in
	
	def show_GS(self, zernike, amplitude):
		input_phase = self.zernike_to_phase(zernike, amplitude)
		reading = self.forward(self.phase_to_pupil(input_phase))
		retrieved = self.GS(reading.intensity, niter=10)
		fig, axs = plt.subplots(1, 3)
		fig.suptitle(f"Lantern phase retrieval, Zernike {zernike}, amplitude {amplitude}")
		fig.subplots_adjust(top=1.4, bottom=0.0)
		for ax in axs:
			ax.set_xticks([])
			ax.set_yticks([])
		axs[0].imshow(input_phase.shaped)
		axs[0].set_title("Phase screen")
		axs[1].imshow(np.abs(reading.intensity.shaped))
		axs[1].set_title("Lantern output")
		axs[2].imshow(retrieved.phase.shaped)
		axs[2].set_title("Retrieved phase")
		plt.show()
  
	def plot_outputs(self):
		fig, axs = plt.subplots(5, 4)
		plt.suptitle("Photonic lantern entrance modes")
		plt.subplots_adjust(wspace=-0.7, hspace=0.1)
		for (i, o) in enumerate(self.outputs):
			r, c = i // 4, i % 4
			axs[r][c].imshow(np.abs(self.input_to_2d(o)))
			axs[r][c].set_xticks([])
			axs[r][c].set_yticks([])
			# plt.axis('off')
		# plt.axis('off')
		fig.delaxes(axs[-1][-1])
		plt.savefig(PROJECT_ROOT + "/figures/lantern_modes_19.png", dpi=600, bbox_inches="tight")
		plt.show()