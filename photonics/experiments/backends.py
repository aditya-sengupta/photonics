# defines "photonic backends"
# i.e. DM-WFS
# the children yearn for the Julia type tree
# actually why don't I just port this whole thing to Julia? DAO has an interface for it...
# because it's too much work to rewrite everything

from abc import ABC

class PhotonicBackend(ABC):
    """
    - get intensities off an image given that we've got centroids and a radius
    - measure darks and PL flats and save them appropriately
    - take an interaction matrix and invert it
    - make linearity curves
    - do a "pseudo-closed-loop" and a real closed loop
    """
    def set_centroids(self, centroids, image, save=True):
        """
        Sets the centroids of each PL port based on 
        - an initial guess of each position, in `centroids`
        - a reference image, in `image`.
        """
        self.xc, self.yc = ports_in_radial_order(refine_centroids(centroids))
        self.yi, self.xi = np.indices(image.shape)
        self.masks = [(self.xi - xc) ** 2 + (self.yi - yc) ** 2 <= self.spot_radius_px ** 2 for (xc, yc) in zip(self.xc, self.yc)]
        self.centroids_dt = datetime_now()
        if save:
            new_centroids = np.zeros_like(centroids)
            new_centroids[:,0] = self.xc
            new_centroids[:,1] = self.yc
            with h5py.File(self.filepath(f"centroids_{self.centroids_dt}", ext="hdf5"), "w") as f:
                c = f.create_dataset("centroids", data=new_centroids)
                c.attrs["spot_radius_px"] = self.spot_radius_px
                
    def get_intensities(self, img, exclude=[]):
        intensities = np.zeros(self.Nports - len(exclude))
        j = 0
        for i in range(self.Nports - len(exclude)):
            while j in exclude:
                j += 1
            intensities[i] = np.sum(img[self.masks[i]])
            j += 1

        return normalize(intensities)
    
    def reconstruct_image(self, img, intensities):
        recon_image = np.zeros_like(img)
        for (i, intensity) in enumerate(intensities):
            recon_image[self.masks[i]] = intensity

        return recon_image
    
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
    
class SimBackend(PhotonicBackend):
    pass

class muirSEALBackend(PhotonicBackend):
    pass