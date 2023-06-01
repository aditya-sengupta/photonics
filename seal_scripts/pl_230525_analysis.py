# %%
import os, sys, re, tqdm
sys.path.append("..")
from src.lantern_reader import LanternReader

reader = LanternReader(
    nports = 18,
    cutout_size = 20,
    fwhm = 12,
    ext = "png",
    imgshape = (1200, 1920),
    subdir="pl_230525"
)

img = reader.read_image("pl_230525_z3_a0.0")
reader.set_centroids(img, min_energy=30000)
plt.imshow(img)
plt.show()
plt.imshow(reader.reconstruct_image(img, reader.get_intensities(img)))

# %%
f = os.listdir(reader.directory)
pattern = re.compile("pl_230525_z(\d+)(_a|_)(.+).png")
f2 = list(filter(lambda x: x is not None, map(lambda x: pattern.match(x), f)))
zerns, amps, intensities = np.zeros(len(f2)), np.zeros(len(f2)), np.zeros((len(f2), reader.nports))
for (j, m) in enumerate(tqdm.tqdm(f2)):
    img = reader.read_image(m[0][:-4])
    zerns[j] = int(m[1])
    amps[j] = float(m[3])
    intensities[j] = reader.get_intensities(img)

# %%
reader.save("230525_zerns", zerns)
reader.save("230525_amps", amps)
reader.save("230525_intensities", intensities)

# %%
def plot_intensities_for_zernike(zern):
    inds = np.where(zerns == zern)
    z_amps, z_ints = amps[inds], intensities[inds]
    sortperm = np.argsort(z_amps)
    z_amps, z_ints = z_amps[sortperm], z_ints[sortperm]
    for i in range(reader.nports):
        plt.plot(z_amps, z_ints[:,i], c=plt.cm.RdBu(i / reader.nports), label=(f"Port {i+1}" if i%3 == 2 else None))
    plt.xlabel("Input amplitude (DM P2V)")
    plt.ylabel("Port intensity (averaged counts)")
    plt.title(f"SMF outputs for Zernike {zern}")
    plt.legend()

# %%
for z in range(2, 20):
    plot_intensities_for_zernike(z)
    plt.show()

# %%
