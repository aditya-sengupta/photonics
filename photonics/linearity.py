# Common linearity tools so I don't keep rewriting them.

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

def interpolate_weights(arr, x):
	"""
	Given a sorted array arr and some arr[0] < x < arr[-1], finds the left index idx and the left weight w such that x = w * arr[idx] + (1 - w) * arr[idx+1].
	
	Arguments:
	arr - np.ndarray, (n,)
		The sorted array to interpolate.
	x - float
		The value whose interpolation we want.
		
	Returns:
	idx - int
		The left-hand index.
	w - float
		The weight to apply to the left-hand index.
	"""
	idx = np.where(arr > x)[0][0] - 1
	w = 1 - (x - arr[idx]) / (arr[idx+1] - arr[idx])
	return idx, w

def make_interaction_matrix(amplitudes, mode_sweep, poke_amplitude=0.1):
	"""
	Makes a modal interaction matrix given a linearity sweep dataset.
	
	Arguments:
	amplitudes - np.ndarray, (namp,)
		The amplitudes poked for each Zernike mode.
	mode_sweep - np.ndarray, (nzern, namp, nwfs,)
		The WFS response for each Zernike number and amplitude.
	poke_amplitude - float
		The poke amplitude used for the push/pull interaction. If this isn't in "amplitudes", the given readings will be interpolated.
		
	Returns:
	interaction_matrix - np.ndarray, (nwfs, nzern,)
		The interaction matrix.
	"""
	nzern, namp, nwfs = mode_sweep.shape
	assert len(amplitudes.shape) == 1 and len(amplitudes) == namp, "Malformed input; make sure the amplitude arrays match."
	interaction_matrix = np.zeros((nwfs, nzern))
	idx_pos, w_pos = interpolate_weights(amplitudes, poke_amplitude)
	idx_neg, w_neg = interpolate_weights(amplitudes, -poke_amplitude)
	for i in range(nzern):
		s_push = w_pos * mode_sweep[i,idx_pos,:] + (1 - w_pos) * mode_sweep[i,idx_pos+1,:]
		s_pull = w_neg * mode_sweep[i,idx_neg,:] + (1 - w_neg) * mode_sweep[i,idx_neg+1,:]
		s = (s_push - s_pull) / (2 * poke_amplitude)
		interaction_matrix[:,i] = s.ravel()
	
	return interaction_matrix

def make_linearity(amplitudes, mode_sweep, cm):
	"""
	Makes a linearity matrix (response in Zernike space) given a command matrix and amplitudes.
	
	Arguments:
	amplitudes - np.ndarray, (namp,)
		The amplitudes poked for each Zernike mode.
	mode_sweep - np.ndarray, (nzern, namp, nwfs,)
		The WFS response for each Zernike number and amplitude.
	cm - np.ndarray, (nzern, nwfs,)
		The command matrix.
	"""
	nzern, nwfs = cm.shape
	namp = len(amplitudes)
	responses = np.zeros((nzern, namp, nzern))
	idx_zero, w_zero = interpolate_weights(amplitudes, 0.0)
	flat = w_zero * mode_sweep[0,idx_zero,:] + (1 - w_zero) * mode_sweep[0,idx_zero+1,:]
	for z in trange(nzern):
		for j in range(len(amplitudes)):
			flattened = mode_sweep[z,j,:] - flat
			response = cm @ flattened
			responses[z,j,:] = response
	
	return responses

def plot_linearity(amplitudes, responses, title_mod="", zlabels=None):
	nzern = responses.shape[0]
	if zlabels is None:
		zlabels = [f"Z{i+1}" for i in range(nzern)]
	nrow = 3
	ncol = int(np.ceil(nzern / 3))
	fig, axs = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(3 * nrow, 3 * ncol))
	title = "Photonic lantern linearity curves"
	if title_mod != "":
		title += f", {title_mod}"
	plt.suptitle(title, y=1.03)
	for i in range(nzern):
		r, c = i // 3, i % 3
		if ncol > 1:
			ax = axs[r][c]
		else:
			ax = axs[c]
		ax.set_ylim([min(amplitudes), max(amplitudes)])
		ax.title.set_text(zlabels[i])
		ax.plot(amplitudes, amplitudes, '--k')
		for j in range(nzern):
			alpha = 1 if i == j else 0.1
			ax.plot(amplitudes, responses[i,:,j], alpha=alpha)
	for k in range(i + 1, nrow * ncol):
		r, c = k // 3, k % 3
		fig.delaxes(axs[r][c])
	plt.show()
 
def linearity_loss(amplitudes, linearity_responses):
    comparison = np.eye(linearity_responses.shape[2])[:, np.newaxis, :]
    return np.sum((comparison * amplitudes[np.newaxis, :, np.newaxis] - linearity_responses) ** 2)
