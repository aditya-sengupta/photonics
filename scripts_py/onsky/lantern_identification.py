from photonics.experiments.shane import ShaneLantern
import numpy as np
from matplotlib import pyplot as plt

lant = ShaneLantern()

def gaussian_2d(x, y, xc, yc, r):
	return np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * r**2))

def hexagon_pattern(nrings, core_offset):
	nports = 1 + 3 * nrings * (nrings - 1)
	port_positions = np.zeros((nports, 2))
	nports_so_far = 0
	for i in range(nrings):
		nports_per_ring = max(1,6*i)
		theta = 0
		current_position = i * core_offset * np.array([np.cos(theta), np.sin(theta)])
		next_position = i * core_offset * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi/3)])
		for j in range(nports_per_ring):
			if i > 0 and j % i == 0:
				theta += np.pi / 3
				current_position = next_position
				next_position = i * core_offset * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
			cvx_coeff = 0 if i == 0 else (j % i) / i
			port_positions[nports_so_far,:] = (1 - cvx_coeff) * current_position + cvx_coeff * next_position
			nports_so_far += 1
	return port_positions

i_iter = 1
def onclick(event):
	global ix, iy, i_iter
	ix, iy = event.xdata, event.ydata
	#print(ix, iy)

	if ix != None:
		print('Click %d: x = %d, y = %d'%(i_iter, ix, iy))
		i_iter += 1
	elif ix == None:
		print(ix, iy)

	global coords
	coords.append((ix, iy))

	if ix == None and iy == None:
		fig.canvas.mpl_disconnect(cid)
		plt.close()

	return coords

pattern = hexagon_pattern(3, 20) * 5 + 250
test_image = sum(gaussian_2d(lant.xi, lant.yi, xc, yc, 20) for (xc, yc) in pattern)
accepted_positions = False
while not accepted_positions:
	test_image_ret = test_image
	fig = plt.figure()
	plt.imshow(test_image_ret)
	plt.title("Click on all the PL ports!")
	coords = []
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()

	coords = np.array(coords)
	plt.imshow(test_image_ret, label="True coordinates")
	plt.scatter(*coords.T, label="Clicked coordinates")
	plt.legend()
	plt.show()
	
	accepted_positions = input("Identified PL ports OK? [y/n] ") == "y"

lant.set_centroids(np.array(coords))
accepted_radius = False
while not accepted_radius:
	fig, axs = plt.subplots(3,1, figsize=(3,9))
	for ax in axs:
		ax.set_axis_off()
	axs[0].imshow(test_image_ret)
	axs[1].imshow(sum(test_image_ret * m for m in lant.masks))
	axs[2].imshow(lant.reconstruct_image(test_image_ret, lant.get_intensities(test_image_ret)))
	plt.show()
	
	update_radius = input("Reconstructed PL image OK? [y/new radius] ")
	if update_radius != "y":
		lant.spot_radius_px = float(update_radius)
		lant.set_centroids(lant.centroids)
	else:
		accepted_radius = True
	