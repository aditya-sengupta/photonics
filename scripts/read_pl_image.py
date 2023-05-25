# %%
import sys
sys.path.append("..")
from src.lantern_reader import LanternReader

if __name__ == "__main__":
    reader = LanternReader(
        nports = 18,
        fwhm = 15.0,
        ext = "png",
        imgshape = (1200, 1920)
    )

    img = reader.read_image("first_lantern_test")
    reader.set_centroids(img)
    intensities = reader.get_intensities(img)
    plt.imshow(img, cmap='magma')
    plt.show()
    plt.imshow(reader.reconstruct_image(img, intensities), cmap='magma')
    plt.show()

# %%
plt.scatter(reader.xc, reader.yc)
plt.plot(reader.xc, reader.yc)

# %%
