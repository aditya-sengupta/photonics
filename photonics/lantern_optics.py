import os
from os.path import join
import numpy as np
import hcipy as hc
import lightbeam as lb
from hcipy import imshow_field
from matplotlib import pyplot as plt
from tqdm import trange
from .utils import PROJECT_ROOT, date_now

class LanternOptics:
	def __init__(self, optics, f_number):
		self.nports = 19 # update this later
		self.nmodes = 9 # update this later as well
		self.f_number = f_number
		mesh_extent, mesh_spacing = optics.mesh_extent, optics.mesh_spacing
		self.telescope_diameter = optics.telescope_diameter
		self.wl = 1.55
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
		self.load_outputs()
		self.make_command_matrix(optics, self.nmodes)

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
		outputs = np.load(PROJECT_ROOT + "/data/backprop_19.npy")
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

	def show_lantern_output(self, focal_field):
		coeffs = self.lantern_coeffs(focal_field)
		lantern_reading = np.abs(sum(c * lf for (c, lf) in zip(coeffs, self.plotting_launch_fields))) ** 2
		fig, axs = plt.subplots(1, 3)
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
		
	def make_command_matrix(self, optics, rerun=True):
		dm = optics.deformable_mirror
		dm.flatten()
		self.image_ref = np.abs(self.lantern_coeffs(optics.focal_propagator(dm.forward(optics.wf)))) ** 2
		cmd_path = join(PROJECT_ROOT, "data", "secondstage_lantern", f"cm_{date_now()}_{optics.dm_basis}.npy")
		if (not rerun) and os.path.exists(cmd_path):
			self.command_matrix = np.load(cmd_path)
		else:
			probe_amp = 0.01 * self.wl / (4 * np.pi)
			num_modes = self.nmodes
			slopes = []

			for ind in range(num_modes):
				slope = 0

				# Probe the phase response
				for s in [1, -1]:
					amp = np.zeros((dm.num_actuators,))
					amp[ind] = s * probe_amp
					dm.actuators = amp
					dm_wf = dm.forward(optics.wf)
					image = self.lantern_coeffs(optics.focal_propagator(dm_wf))
					slope += s * (image - self.image_ref) / (2 * probe_amp)

				slopes.append(slope)

			dm.flatten()
			slopes = hc.ModeBasis(slopes)
			self.command_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=1e-3, svd=None)
			np.save(join(PROJECT_ROOT, "data", "secondstage_lantern", f"cm_{date_now()}_{optics.dm_basis}.npy"), self.command_matrix)
  
	def make_linearity(self, optics, lim=0.1, step=None):
		dm = optics.deformable_mirror
		if step is None:
			step = lim / 4
		amplitudes = np.arange(-lim, lim*1.001, step)
		linearity_responses = np.zeros((self.nmodes, len(amplitudes), self.nmodes))
		for z in range(self.nmodes):
			for (j,a) in enumerate(amplitudes):
				dm.actuators[z] = a * self.wl / (4 * np.pi)
				focal_wavefront = optics.focal_propagator(dm.forward(optics.wf))
				flattened = np.abs(self.lantern_coeffs(focal_wavefront)) ** 2 - self.image_ref
				response = self.command_matrix @ flattened
				linearity_responses[z,j,:] = response
			dm.flatten()
		
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
		return hc.Wavefront(hc.Field(reading.ravel(), self.focal_grid), wavelength=self.wl*1e-6)
	
	def backward(self, reading):
		coeffs = self.lantern_reverse @ reading.electric_field
		lantern_input = self.input_to_2d(coeffs @ self.outputs, zoomed=False)
		focal_wf = hc.Wavefront(hc.Field(lantern_input.ravel(), self.focal_grid), wavelength=self.wl*1e-6)
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
		for _ in trange(niter):
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
	
	def show_GS(self, zernike, amplitude, niter=10):
		input_phase = self.zernike_to_phase(zernike, amplitude)
		reading = self.forward(self.phase_to_pupil(input_phase))
		retrieved = self.GS(reading.intensity, niter=niter)
		retphase = retrieved.phase # (retrieved.phase % np.pi) - np.pi / 2
		fig, axs = plt.subplots(1, 3)
		fig.suptitle(f"Lantern phase retrieval, Zernike {zernike}, amplitude {amplitude}")
		fig.subplots_adjust(top=1.4, bottom=0.0)
		for ax in axs:
			ax.set_xticks([])
			ax.set_yticks([])
		# vmin = np.minimum(np.min(input_phase), np.min(retphase))
		# vmax = np.maximum(np.max(input_phase), np.max(retphase))
		axs[0].imshow(input_phase.shaped)
		axs[0].set_title("Phase screen")
		axs[1].imshow(np.abs(reading.intensity.shaped))
		axs[1].set_title("Lantern output")
		hc.imshow_field(retphase * self.aperture, ax=axs[2])
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
		plt.savefig(join(PROJECT_ROOT, "figures", "lantern_modes_19.png"), dpi=600, bbox_inches="tight")
		plt.show()
  