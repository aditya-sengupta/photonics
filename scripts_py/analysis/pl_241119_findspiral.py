# %%
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import trange
from photonics.experiments.shane import ShaneLantern
# %%
with h5py.File("../../data/pl_241119/centroids_2024-11-19T18.19.22.185917.hdf5") as f:
    centroids = np.array(f["centroids"])
# %%
lant = ShaneLantern()
lant.set_centroids(centroids)

# %%
spiral_search_runs = [
    # "stability_2024-11-19T18.35.31.751850.hdf5",
    """    "stability_2024-11-19T18.49.20.552640.hdf5",
    "stability_2024-11-19T18.52.19.198809.hdf5",
    "stability_2024-11-19T18.52.28.898958.hdf5",
    "stability_2024-11-19T18.53.36.906790.hdf5",
    "stability_2024-11-19T18.53.46.656923.hdf5",
    "stability_2024-11-19T18.54.44.361814.hdf5",
    "stability_2024-11-19T18.57.36.228830.hdf5",
    "stability_2024-11-19T19.02.49.030057.hdf5",
    "stability_2024-11-19T19.06.03.842302.hdf5",
    "stability_2024-11-19T19.08.08.360465.hdf5",
    "stability_2024-11-19T19.13.02.699090.hdf5","""
# %%
spiral_search_runs = [
    "stability_2024-11-19T21.08.09.370403.hdf5"
]
# %%
with h5py.File(f"../../data/pl_241119/stability_2024-11-19T18.35.31.751850.hdf5") as f:
    ref_image = np.array(f["pl_images"][0]) / 2
# %%
for x in spiral_search_runs:
    with h5py.File(f"../../data/pl_241119/{x}") as f:
        exp_ms = f["pl_intensities"].attrs["exp_ms"]
        ref_image_conversion = exp_ms / 500
        images = np.array(f["pl_images"])
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        im = ax.imshow(ref_image * ref_image_conversion, cmap='viridis')

        def update(frame):
            im.set_array((images[frame, :, :] - ref_image * ref_image_conversion))
            # (np.sum(lant.masks, axis=0)
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(images), blit=True)
        ani.save(f'animation_{x}.mp4', writer='ffmpeg')
        plt.show()
        plt.plot(np.array([lant.get_intensities(img - ref_image * ref_image_conversion) for img in images]))
# %%
