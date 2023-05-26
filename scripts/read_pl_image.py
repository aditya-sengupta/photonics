# %%
import sys
sys.path.append("..")
from src.lantern_reader import LanternReader

guesses = np.array([[117, 782],
    [178, 993],
    [304, 1171],
    [334, 715],
    [391, 931],
    [440, 408],
    [508, 1239],
    [529, 608],
    [582, 824],
    [586, 1037],
    [655, 401],
    [771, 918],
    [844, 504],
    [902, 1095],
    [990, 895],
    [255, 510],
    [770, 710],
    [970, 680]])

if __name__ == "__main__":
    reader = LanternReader(
        nports = 18,
        cutout_size = 25,
        fwhm = 15,
        ext = "png",
        imgshape = (1200, 1920),
        guess_positions=guesses
    )

    img_ref = reader.read_image("first_lantern_test")
    img = reader.read_image("pl_flat_230525_1326")
    reader.set_centroids(img_ref)
    intensities = reader.get_intensities(img)
    plt.imshow(img, cmap='magma')
    plt.show()
    plt.imshow(reader.reconstruct_image(img, intensities), cmap='magma')
    plt.show()

# %%
plt.scatter(reader.xc, reader.yc, c=np.array([0] + [0.5] * 6 + [1.0] * 11))
plt.xticks([])
plt.yticks([])
for (i, (xc, yc)) in enumerate(zip(reader.xc, reader.yc)):
    plt.annotate(i + 1, (xc, yc), xytext=(xc+5, yc+5))

plt.savefig("../figures/lantern_order.png")
# %%
