# %%
import numpy as np
from matplotlib import pyplot as plt
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

reference_image = np.zeros((520,656))
xg, yg = np.indices((520,656))

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
def center_of_mass(img, indices_x, indices_y):
    # Returns the center of mass (intensity-weighted sum) in the x and y direction of the image.
    xc = np.sum(img * indices_x) / np.sum(img)
    yc = np.sum(img * indices_y) / np.sum(img)
    return xc, yc

# %%
cutout_size = 12
new_centroids = np.zeros_like(centroids)
for i in range(19):
    xl, xu = int(identified_centroids[i,0]) - cutout_size, int(identified_centroids[i,0]) + cutout_size + 1
    yl, yu = int(identified_centroids[i,1]) - cutout_size, int(identified_centroids[i,1]) + cutout_size + 1
    new_centroids[i] = center_of_mass(reference_image[xl:xu, yl:yu], xg[xl:xu, yl:yu], yg[xl:xu, yl:yu])

plt.imshow(reference_image, cmap='gray')
plt.scatter(new_centroids[:,1], new_centroids[:,0], s=1)
# %%
print(float(np.std(identified_centroids - centroids)))
print(float(np.std(new_centroids - centroids)))
# %%
