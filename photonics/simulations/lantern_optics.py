from os.path import join
import numpy as np
import hcipy as hc
import lightbeam as lb
from hcipy import imshow_field
from matplotlib import pyplot as plt
from ..utils import PROJECT_ROOT, date_now, zernike_names, nanify
from .command_matrix import make_command_matrix

class LanternOptics:
	"""
	Implements the optics around a photonic lantern and wraps propagations from Lightbeam to determine the lantern's behaviour.
 
	Need to ask Emiel about the canonical hcipy way of organizing this vs. the parent "optics" class.
	"""
	def __init__(self, optics, nports=19, nmodes=9):
		self.name = f"lantern_ports{nports}_modes{nmodes}"
		self.nports = nports # update this later
		self.nmodes = nmodes # update this later as well
		self.f_number = optics.lantern_fnumber
		mesh_extent, mesh_spacing = optics.mesh_extent, optics.mesh_spacing
		self.telescope_diameter = optics.telescope_diameter
		self.wl = optics.wl
		self.final_scale = 8 # tapering factor of lantern
		self.cladding_radius = 37/2 * self.final_scale
		self.core_offset = self.cladding_radius / 2.5 # offset of cores from origin
		self.nclad = 1.4504
		self.ncore = self.nclad + 0.01036 # lantern core refractive index
		self.njack = np.sqrt(self.nclad**2-0.125**2) # jacket index
		self.rcore = 2.2
		self.z_ex = 60_000
		self.lant = lb.optics.make_lant19(self.core_offset,self.rcore,self.cladding_radius,0,self.z_ex, (self.ncore,self.nclad,self.njack),final_scale=1/self.final_scale)
		self.make_mesh(mesh_extent, mesh_spacing)
		self.launch_fields = [
			lb.normalize(lb.lpfield(self.xg-pos[0], self.yg-pos[1], 0, 1, self.rcore, self.wl*1e6, self.ncore, self.nclad))
			for pos in self.lant.init_core_locs
		]
		self.plotting_launch_fields = [
			lb.normalize(lb.lpfield(self.xg-pos[0], self.yg-pos[1], 0, 1, 5 * self.rcore, self.wl*1e6, self.ncore, self.nclad))
			for pos in self.lant.init_core_locs
		]
		self.lbprop = lb.Prop3D(self.wl*1e6, self.mesh, self.lant, self.nclad)
		self.lantern_basis = np.array([lf.ravel() for lf in self.launch_fields]).T
		self.lantern_reverse = np.linalg.inv(self.lantern_basis.T @ self.lantern_basis) @ self.lantern_basis.T
		out = np.zeros_like(self.xg)
		self.lant.set_IORsq(out, self.z_ex)
		# out = out[self.mesh.PML:-self.mesh.PML,self.mesh.PML:-self.mesh.PML]
		self.input_footprint = np.where(out[self.mesh.PML:-self.mesh.PML,self.mesh.PML:-self.mesh.PML] >= self.nclad ** 2)
		self.complement_mask = np.ones_like(out, dtype=bool)
		for (i, j) in zip(self.input_footprint[0], self.input_footprint[1]):
			self.complement_mask[i,j] = False
		self.extent_x = (np.min(self.input_footprint[0]), np.max(self.input_footprint[0]))
		self.extent_y = (np.min(self.input_footprint[1]), np.max(self.input_footprint[1]))
		self.load_outputs()
		self.focal_propagator = optics.focal_propagator
		self.focal_grid = optics.focal_grid
		self.input_ref = optics.im_ref
		make_command_matrix(optics.deformable_mirror, self, optics.wf, probe_amp=1.2e-8, rerun=True)

	def input_to_2d(self, input_efield, zoomed=True, restore_outside=False):
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
		if restore_outside and not zoomed:
			input_efield_2d[self.complement_mask] = self.input_ref.electric_field.shaped[self.complement_mask]
		return input_efield_2d
		
	def make_mesh(self, mesh_extent, mesh_spacing):
		PML = 8
		self.mesh = lb.RectMesh3D(
			xw = mesh_extent, # um
			yw = mesh_extent, # um
			zw = self.z_ex, # um
			ds = mesh_spacing, # um
			dz = 50, # um
			PML = PML # grid units
		)
		self.lant.set_sampling(self.mesh.xy)
		xg, yg = self.mesh.grids_without_pml()
		self.w = self.mesh.xy.get_weights()[PML:-PML,PML:-PML]
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
		
		np.save(PROJECT_ROOT + f"/data/backprop_19_{date_now()}.npy", np.array(outputs))
		np.save(PROJECT_ROOT + "/data/backprop_19.npy", np.array(outputs))
		
	def load_outputs(self):
		outputs = np.load(join(PROJECT_ROOT, "data", "backprop_19.npy"))
		self.outputs = np.array([self.sanitize_output(x) for x in outputs])
		self.projector = np.linalg.inv(self.outputs @ self.outputs.T) @ self.outputs

	def lantern_coeffs(self, focal_field):
		profile_to_project = focal_field.electric_field.shaped[self.input_footprint]
		return self.projector @ profile_to_project

	def lantern_output(self, focal_field):
		coeffs = self.lantern_coeffs(focal_field)
		# projected = self.input_to_2d(coeffs @ self.outputs)
		lantern_reading = sum(c * lf for (c, lf) in zip(coeffs, self.launch_fields))
		return coeffs, lantern_reading

	def plotting_lantern_output(self, focal_field):
		coeffs = self.lantern_coeffs(focal_field)
		lantern_reading = sum(c * lf for (c, lf) in zip(coeffs, self.plotting_launch_fields))
		return lantern_reading

	def readout(self, wf):
		return np.abs(
			self.lantern_coeffs(self.focal_propagator(wf))
		) ** 2

	def show_lantern_output(self, focal_field):
		coeffs = self.lantern_coeffs(focal_field)
		lantern_reading = np.abs(sum(c * lf for (c, lf) in zip(coeffs, self.plotting_launch_fields))) ** 2
		fig, axs = plt.subplots(1, 2)
		fig.suptitle("Photonic lantern response")
		fig.subplots_adjust(top=1.4, bottom=0.0)
		for ax in axs:
			ax.set_xticks([])
			ax.set_yticks([])
		imshow_field(focal_field.intensity, ax=axs[0])
		axs[0].set_title("Lantern input")
		axs[1].imshow(np.abs(lantern_reading))
		axs[1].set_title("Lantern output")
		plt.show()
  
	def make_linearity(self, optics, lim=0.1, step=None):
		conversion = (4 * np.pi / self.wl)
		dm = optics.deformable_mirror
		if step is None:
			step = lim / 4
		amplitudes = np.arange(-lim, lim*1.001, step)
		linearity_responses = np.zeros((self.nmodes, len(amplitudes), self.nmodes))
		dm.flatten()
		for ind in range(self.nmodes):
			for (j, a) in enumerate(amplitudes):
				amp = np.zeros((dm.num_actuators,))
				amp[ind] = a / conversion
				dm.actuators = amp
				lantern_image = self.readout(dm.forward(optics.wf))
				flattened = lantern_image - self.image_ref
				response = self.command_matrix.dot(flattened)[:self.nmodes]
				linearity_responses[ind,j,:] = response * conversion
			dm.flatten()
		
		return amplitudes, linearity_responses

	def show_linearity(self, amplitudes, linearity_responses):
		nzern = linearity_responses.shape[0]
		fig, axs = plt.subplots(int(np.ceil(nzern // 3)), 3, sharex=True, sharey=True, figsize=(9, 9))
		plt.suptitle(f"Photonic lantern linearity curves (rad), f/{self.f_number}")
		for i in range(nzern):
			r, c = i // 3, i % 3
			axs[r][c].set_ylim([min(amplitudes), max(amplitudes)])
			axs[r][c].title.set_text(zernike_names[i])
			for j in range(nzern):
				alpha = 1 if i == j else 0.3
				axs[r][c].plot(amplitudes, linearity_responses[i,:,j], alpha=alpha)
		plt.show()
		
	def forward(self, optics, pupil):
		focal = self.focal_propagator.forward(pupil)
		_, reading = self.lantern_output(focal)
		return hc.Wavefront(hc.Field(reading.ravel(), optics.focal_grid), wavelength=self.wl)
	
	def backward(self, optics, reading, restore_outside=False):
		coeffs = self.lantern_reverse @ reading.electric_field
		lantern_input = self.input_to_2d(coeffs @ self.outputs, zoomed=False, restore_outside=restore_outside)
		focal_wf = hc.Wavefront(hc.Field(lantern_input.ravel(), self.focal_grid), wavelength=self.wl)
		pupil_field = self.focal_propagator.backward(focal_wf)
		return pupil_field

	def GS_init(self, optics, img, guess=None, restore_outside=False):
		# Normalization
		img /= img.sum()
		
		# Amplitude in - measured (or assumed to be known)
		measuredAmplitude_in = optics.wf
		# Amplitude out - measured
		measuredAmplitude_out = np.sqrt(img) 
		# Forward Propagation in WFS assuming 0 phase
		EM_out = self.forward(optics, measuredAmplitude_in)
		# replacing by known amplitude
		phase_out_0 = EM_out.phase
		EM_out_replaced = measuredAmplitude_out * np.exp(1j*phase_out_0)
		# Back Propagation in PL
		if guess is None:
			EM_in = self.backward(optics, hc.Wavefront(EM_out_replaced, wavelength=self.wl), restore_outside=restore_outside)
		else:
			EM_in = guess
		filteredAmplitude_in = np.abs(self.backward(optics, EM_out).electric_field)
		return EM_in, filteredAmplitude_in, measuredAmplitude_out

	def GS_iteration(self, optics, EM_in, measuredAmplitude_in, measuredAmplitude_out, restore_outside=False):
		# replacing by known amplitude
		phase_in_k = EM_in.phase
		EM_in = hc.Wavefront(measuredAmplitude_in*np.exp(1j*phase_in_k),wavelength=self.wl)
		# Lantern forward propagation
		EM_out = self.forward(optics, EM_in)
		# replacing by known amplitude
		phase_out_k = EM_out.phase
		EM_out = hc.Wavefront(measuredAmplitude_out*np.exp(1j*phase_out_k), wavelength=self.wl)
		# Lantern backward propagation    
		EM_in = self.backward(optics, EM_out, restore_outside=restore_outside)
			
		return EM_in

	def GS(self, optics, img, guess=None, niter=10):
		"""
		Gerchberg-Saxton algorithm
		"""
		EM_in, measuredAmplitude_in, measuredAmplitude_out = self.GS_init(optics, img, guess)
		for _ in range(niter):
			EM_in = self.GS_iteration(optics, EM_in, measuredAmplitude_in, measuredAmplitude_out)
			
		return EM_in
	
	def show_GS(self, optics, zernike, amplitude, guess=None, niter=10):
		input_phase = optics.zernike_to_phase(zernike, amplitude)
		input_pupil = optics.phase_to_pupil(input_phase)
		reading = self.forward(optics, input_pupil)
		input_focal = self.focal_propagator.forward(input_pupil)
		coeffs = self.lantern_coeffs(input_focal)
		out_to_plot = np.abs(sum(c * lf for (c, lf) in zip(coeffs, self.plotting_launch_fields))) ** 2
		retrieved = self.GS(optics, reading.intensity, guess=guess, niter=niter)
		retphase = (retrieved.phase % np.pi) - np.pi / 2
		fig, axs = plt.subplots(1, 3)
		fig.suptitle(f"Lantern phase retrieval, Zernike {zernike}, amplitude {amplitude}")
		fig.subplots_adjust(top=1.4, bottom=0.0)
		for ax in axs:
			ax.set_xticks([])
			ax.set_yticks([])
		vmin = np.minimum(np.min(input_phase), np.min(retphase))
		vmax = np.maximum(np.max(input_phase), np.max(retphase))
		axs[0].imshow(nanify(input_phase.shaped, optics.aperture.shaped), cmap="RdBu", vmin=vmin, vmax=vmax)
		axs[0].set_title("Phase screen")
		axs[1].imshow(out_to_plot)
		axs[1].set_title("Lantern output")
		axs[2].imshow(nanify(retphase.shaped, optics.aperture.shaped), cmap="RdBu", vmin=vmin, vmax=vmax)
		axs[2].set_title("Retrieved phase")
		plt.show()
  
	def plot_outputs(self):
		rm, cm = 4, 5
		fig, axs = plt.subplots(rm, cm)
		plt.suptitle("Photonic lantern entrance modes")
		plt.subplots_adjust(wspace=0.05, hspace=0.05)
		for (i, o) in enumerate(self.outputs):
			r, c = i // cm, i % cm
			axs[r][c].imshow(np.abs(self.input_to_2d(o)))
			axs[r][c].set_xticks([])
			axs[r][c].set_yticks([])
			# plt.axis('off')
		# plt.axis('off')
		fig.delaxes(axs[-1][-1])
		plt.savefig(join(PROJECT_ROOT, "figures", "lantern_modes_19.png"), dpi=600, bbox_inches="tight")
		plt.show()
  