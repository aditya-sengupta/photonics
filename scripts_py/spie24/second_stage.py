import numpy as np
import hcipy as hc
import sys
from copy import copy
from tqdm import tqdm
from .wfs_filter import HighPassFilter, LowPassFilter

# BREAKING THIS 2024-12-15

def correction(
	optics, pyramid, lantern, 
	ncpa=None, f_cutoff=30,
 	f_loop=800, num_iterations=200, gain=0.3, leakage=0.999, 
	second_stage_iter=100,
	use_pyramid=False, use_lantern=False,
	pyramid_recon="linear", lantern_recon="linear",
):
	"""
	Simulates a full two-stage AO loop.
 
	Switches reconstruction strategies on the pyramid and lantern according to
	pyramid_recon : "perfect" or "linear"
	lantern_recon : "perfect", "linear", "nn", "gs"
	"""
	correction_results = {
		"phases_for" : [],
		"wavefronts_after_dm" : [],
		"pyramid_readings" : [],
		"dm_commands" : [],
		"point_spread_functions" : [],
		"strehl_ratios" : [],
		"focal_zernikes_truth" : [],
		"lantern_readings" : [],
		"lpf_readings" : [],
		"hpf_readings" : []
	}
	a = np.exp(-2 * np.pi * f_cutoff / f_loop)
	pyramid_filter = HighPassFilter(lantern.nmodes, a)
	lantern_filter = LowPassFilter(lantern.nmodes, a)
	layer, dm = optics.layer, optics.deformable_mirror
	layer.reset()
	layer.t = 0
	dm.flatten()
	dt = 1/f_loop
	correction_results["time"] = np.arange(0, dt * num_iterations, dt)
	do_second_stage = False
	GS_output = None
	with tqdm(range(num_iterations), file=sys.stdout) as progress:
		for timestep in progress:
			close_second_stage = use_pyramid and use_lantern and timestep == second_stage_iter
			wf_after_dm = optics.wavefront_after_dm(timestep * dt)
			correction_results["phases_for"].append(layer.phase_for(optics.wl))
			correction_results["wavefronts_after_dm"].append(wf_after_dm.copy())
			if ncpa is not None:
				wf_with_ncpa = hc.Wavefront(
						wf_after_dm.electric_field * ncpa.electric_field,
						wavelength = optics.wl
					)
			else:
				wf_with_ncpa = wf_after_dm
			wf_focal = optics.focal_propagator.forward(wf_with_ncpa)
			correction_results["point_spread_functions"].append(wf_focal.copy())
			strehl_foc = hc.get_strehl_from_focal(wf_focal.intensity/optics.norm, optics.im_ref.intensity/optics.norm)
			correction_results["strehl_ratios"].append(float(strehl_foc))
			dm_command = np.zeros(dm.num_actuators)
			focal_zernikes_truth = optics.zernike_basis.coefficients_for(wf_with_ncpa.phase)
			correction_results["focal_zernikes_truth"].append(focal_zernikes_truth)
			if use_pyramid:
				if pyramid_recon == "perfect":
					pyramid_reading = focal_zernikes_truth * (optics.wl / (4 * np.pi))
				else:
					pyramid_reading = pyramid.reconstruct(wf_after_dm)
				correction_results["pyramid_readings"].append(pyramid_reading)
				hpf_reading = pyramid_filter(pyramid_reading[:pyramid_filter.n])
				correction_results["hpf_readings"].append(hpf_reading)
				dm_command += pyramid_reading
				if do_second_stage:
					dm_command[:pyramid_filter.n] = hpf_reading

			lantern_coeffs = lantern.lantern_coeffs(wf_focal)
			lantern_reading = np.abs(lantern_coeffs) ** 2
			if lantern_recon == "perfect":
				lantern_zernikes_measured = focal_zernikes_truth[:lantern_filter.n] * (optics.wl / (4 * np.pi))
			elif lantern_recon == "nn":
				lantern_zernikes_measured = lantern.nn_reconstruct(lantern_reading) * (optics.wl / (4 * np.pi))
			elif lantern_recon == "gs":
				lantern_zernikes_measured = lantern.gs_inject_recover(np.arange(lantern_filter.n), focal_zernikes_truth[:lantern_filter.n], optics) * (optics.wl / (4 * np.pi))
			else:
				lantern_zernikes_measured = lantern.linear_reconstruct(lantern_reading)
			correction_results["lantern_readings"].append(lantern_zernikes_measured)
			lpf_reading = lantern_filter(lantern_zernikes_measured)
			correction_results["lpf_readings"].append(lpf_reading)
			if use_lantern:
				if do_second_stage:
					dm_command[:lantern_filter.n] += lpf_reading
				elif not use_pyramid:
					dm_command[:lantern.nmodes] += lantern_zernikes_measured

			if close_second_stage:
				tqdm.write(f"Closing photonic lantern loop at iteration {timestep}")
				do_second_stage = True

			correction_results["dm_commands"].append(copy(dm_command))
			dm.actuators = leakage * dm.actuators - gain * dm_command
			strehl_averaged = np.mean(correction_results["strehl_ratios"][max(0,timestep-10):timestep+1])
			progress.set_postfix(strehl=f"{float(strehl_averaged):.3f}")

	return correction_results
