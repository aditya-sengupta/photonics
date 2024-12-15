import numpy as np

"""
- get intensities off an image given that we've got centroids and a radius
- measure darks and PL flats and save them appropriately
- take an interaction matrix and invert it
- make linearity curves
- do a "pseudo-closed-loop" and a real closed loop
"""

@property
def directory(self):
    return path.join(DATA_PATH, self.subdir)

def filepath(self, fname, ext=None):
    """
    The path we want to save/load data from or to.
    """
    if ext is None:
        ext = self.ext
    return path.join(DATA_PATH, self.subdir, fname + "." + ext)

def measure_dark(self, direct=True):
    if direct:
        input("Taking a dark frame, remove the light source!")
    darks = []
    for _ in trange(self.nframes):
        darks.append(self.im.get_data(check=True).astype(float))
        sleep(self.exp_ms / 1000)

    self.dark = np.mean(darks, axis=0)
    self.save(f"dark_exptime_ms_{self.exp_ms}_gain_{self.gain}", self.dark)

def measure_pl_flat(self):
    self.send_zeros(verbose=True)
    self.pl_flat = self.get_intensities(self.get_image())
    self.save(f"pl_flat_{datetime_now()}", self.pl_flat)
    
def plot_ports(self, save=False):
    # only run after set_centroids
    sc = plt.scatter(self.xc, self.yc, c=self.radial_shell)
    plt.xlim((0, self.imgshape[1]))
    plt.ylim((0, self.imgshape[0]))
    plt.xticks([])
    plt.yticks([])
    sc.axes.invert_yaxis()
    for (i, (xc, yc)) in enumerate(zip(self.xc, self.yc)):
        plt.annotate(i + 1, (xc, yc), xytext=(xc+5, yc+5))

    if save:
        plt.savefig(self.filepath(f"port_mask_{datetime_now()}", ext="png"))
    plt.show()
    
def save(self, fname, data, ext="npy", verbose=True):
    fpath = self.filepath(fname, ext=ext)
    if verbose:
        print(f"Saving data to {fpath}")
    np.save(self.filepath(fname, ext=ext), data)
    
# deleted utilities for saturation and ROI; reimplement/pull from Git when needed.
