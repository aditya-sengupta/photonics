from tqdm import trange
from os import path

fpath = "/home/lab/asengupta/photonics/data/pl_250117"

for k in trange(1, 168):
    applySegment(k, 0.2, 0, 0)
    np.save(path.join(fpath, f"{k}_piston.npy"), a.get_data())
    applySegment(k, 0, 0.2, 0)
    np.save(path.join(fpath, f"{k}_tip.npy"), a.get_data())
    applySegment(k, 0, 0, 0.2)
    np.save(path.join(fpath, f"{k}_tilt.npy"), a.get_data())
    applySegment(k, 0, 0, 0)

np.save(path.join(fpath, "flat.npy"), a.get_data())

