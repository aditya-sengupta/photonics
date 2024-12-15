# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py
from photonics.experiments.shane import ShaneLantern
# %%
with h5py.File("../../data/pl_241119/stability_2024-11-19T21.45.55.624215.hdf5") as f:
    images = np.array(f["pl_images"])
    intensities = np.array(f["pl_intensities"])
    print(f["pl_intensities"].attrs["centroids_dt"])
    
# %%
with h5py.File("../../data/pl_241119/centroids_2024-11-19T20.57.43.810016.hdf5") as f:
    centroids = np.array(f["centroids"])

# %%
lant = ShaneLantern()
lant.spot_radius_px = 6
lant.set_centroids(centroids)
plt.plot([lant.get_intensities(images[x]) for x in range(20)]);
# %%
# let's come up with a better centroiding algorithm
# we have approx locations in `centroids`, and a candidate image in `images[0]`
def center_of_mass(img, x_offset, y_offset):
    # Returns the center of mass (intensity-weighted sum) in the x and y direction of the image.
    total = np.sum(img)
    X, Y = np.indices(img.shape)
    x_center = np.sum(X * img) / total
    y_center = np.sum(Y * img) / total
    return x_center + x_offset, y_center + y_offset

cutout_size = 15
new_centroids = np.array([center_of_mass(reference_image[int(yc)-cutout_size:int(yc)+cutout_size, int(xc)-cutout_size:int(xc)+cutout_size], int(xc)-cutout_size, int(yc)-cutout_size) for (xc, yc) in identified_centroids])

for (xc, yc) in identified_centroids:
    plt.imshow(reference_image[int(xc)-cutout_size:int(xc)+cutout_size,int(yc)-cutout_size:int(yc)+cutout_size])
    plt.show()
# %%
plt.scatter(centroids[:,0], centroids[:,1])
plt.scatter(identified_centroids[:,0], identified_centroids[:,1])
plt.scatter(new_centroids[:,0], new_centroids[:,1])
# %%
# it's hard to test if this is working as is...let's make up a test image

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
# %%
centroids = 300 + hexagon_pattern(3, 100)
jitter = np.random.normal(0, 5, size=centroids.shape)
centroids += jitter
radii = np.random.uniform(4, 6, size=19)

reference_image = np.zeros_like(images[0])
xg, yg = np.indices(images[0].shape)

for ((xc, yc), r) in zip(centroids, radii):
    reference_image += np.exp(-((xg - xc)**2 + (yg - yc)**2)/r**2)

reference_image *= 4032 / np.max(reference_image)
reference_image = np.int64(reference_image)
reference_image += np.random.poisson(reference_image)

plt.imshow(reference_image, cmap='gray')
plt.show()
# %%
# now let's say I misidentified the centroids
identified_centroids = np.random.normal(centroids, 3)
plt.imshow(reference_image, cmap='gray')
plt.scatter(identified_centroids[:,1], identified_centroids[:,0])
plt.show()
# %%
cutout_size = 5
new_centroids = np.array([center_of_mass(images[0][int(yc)-cutout_size:int(yc)+cutout_size, int(xc)-cutout_size:int(xc)+cutout_size], int(xc)-cutout_size, int(yc)-cutout_size) for (xc, yc) in identified_centroids])
plt.imshow(reference_image, cmap='gray')
plt.scatter(new_centroids[:,1], new_centroids[:,0])
# %%
