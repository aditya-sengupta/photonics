import numpy as np
import hcipy as hc
import sys
from copy import copy
from tqdm import tqdm
from .wfs_filter import HighPassFilter, LowPassFilter

def correction(
	optics, pyramid, lantern, 
	ncpa=None, f_cutoff=30,
 	f_loop=100, num_iterations=200, gain=0.1, leakage=0.999, 
	use_pyramid=False, use_lantern=False
):
	"""
	Simulates a full two-stage AO loop.
	"""
	correction_results = {
		"phases_for" : [],
		"wavefronts_after_dm" : [],
		"pyramid_readings" : [],
		"dm_commands" : [],
		"dm_shapes" : [],
		"point_spread_functions" : [],
		"strehl_ratios" : []
	}
	a = np.exp(-2 * np.pi * f_cutoff / f_loop)
	pyramid_filter = HighPassFilter(lantern.nmodes, a)
	lantern_filter = LowPassFilter(lantern.nmodes, a)
	layer, dm = optics.layer, optics.deformable_mirror
	layer.reset()
	layer.t = 0
	dt = 1/f_loop
	second_stage_iter = num_iterations // 2
	do_second_stage = False
	do_lantern = (not use_pyramid) and use_lantern
	with tqdm(range(num_iterations), file=sys.stdout) as progress:
		for timestep in progress:
			close_second_stage = use_pyramid and use_lantern and timestep == second_stage_iter
			wf_after_dm = optics.wavefront_after_dm(timestep * dt)
			correction_results["phases_for"].append(layer.phase_for(optics.wl))
			correction_results["wavefronts_after_dm"].append(wf_after_dm.copy())
			correction_results["dm_shapes"].append(copy(dm.surface))
			if ncpa is not None:
				wf_focal = optics.focal_propagator.forward(
					hc.Wavefront(
						wf_after_dm.electric_field * ncpa.electric_field,
						wavelength = optics.wl
					)
				)
			else:
				wf_focal = optics.focal_propagator.forward(wf_after_dm)
			correction_results["point_spread_functions"].append(wf_focal.copy())
			strehl_foc = hc.get_strehl_from_focal(wf_focal.intensity/optics.norm, optics.im_ref.intensity/optics.norm)
			correction_results["strehl_ratios"].append(float(strehl_foc))
			dm_command = np.zeros(dm.num_actuators)
			if use_pyramid:
				pyramid_reading = pyramid.reconstruct(wf_after_dm)
				dm_command += pyramid_reading
				if do_second_stage:
					hpf_reading = pyramid_filter(pyramid_reading[:pyramid_filter.n])
					dm_command[:pyramid_filter.n] = hpf_reading
				correction_results["pyramid_readings"].append(pyramid_reading)

			if do_lantern:
				lantern_zernikes_truth = optics.zernike_basis.coefficients_for(wf_after_dm.phase)
				correction_results["lantern_zernikes_truth"].append(lantern_zernikes_truth)
				lantern_reading = np.abs(lantern.lantern_coeffs(wf_focal)) ** 2
				lantern_zernikes_measured = lantern.command_matrix @ (lantern_reading - lantern.image_ref)
				correction_results["lantern_zernikes_measured"].append(lantern_zernikes_measured)
				if do_second_stage:
					lpf_reading = lantern_filter(lantern_zernikes_measured)
					dm_command[:lantern_filter.n] += lpf_reading
				else:
					dm_command[:lantern.nmodes] += lantern_zernikes_measured

			if close_second_stage:
				tqdm.write(f"Closing photonic lantern loop at iteration {timestep}")
				do_second_stage = True

			correction_results["dm_commands"].append(copy(dm_command))
			dm.actuators = leakage * dm.actuators - gain * dm_command
			strehl_averaged = np.mean(correction_results["strehl_ratios"][max(0,timestep-10):timestep+1])
			progress.set_postfix(strehl=f"{float(strehl_averaged):.3f}")

	return correction_results
