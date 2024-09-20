from photonics.experiments.lantern_reader import LanternReader
import numpy as np
from matplotlib import pyplot as plt

def gaussian_2d(x, y, xc, yc, r):
    return np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * r**2))

def onclick(event):
	global ix, iy
	ix, iy = event.xdata, event.ydata
	#print(ix, iy)

	if ix != None:
		print('Click: x = %d, y = %d'%(ix, iy))
	elif ix == None:
		print(ix, iy)

	global coords
	coords.append((ix, iy))

	if ix == None and iy == None:
		fig.canvas.mpl_disconnect(cid)
		plt.close()

	return coords

x, y = np.indices((520, 626))
image = gaussian_2d(x, y, 300, 100, 10) + gaussian_2d(x, y, 100, 300, 10)
fig = plt.figure()
plt.imshow(image)
coords = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
