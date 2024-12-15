from abc import ABC
import numpy as np
import dao

class LanternCamera(ABC):
    def set_centroids(self, centroids, image, save=False):
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
                
    def refine_centroids(self, centroids, image, cutout_size=12):
        new_centroids = np.zeros_like(centroids)
        for i in range(self.Nports):
            xl, xu = int(centroids[i,0]) - cutout_size, int(centroids[i,0]) + cutout_size + 1
            yl, yu = int(centroids[i,1]) - cutout_size, int(centroids[i,1]) + cutout_size + 1
            new_centroids[i] = center_of_mass(image[xl:xu, yl:yu], xg[xl:xu, yl:yu], yg[xl:xu, yl:yu])
            
        return new_centroids
                
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

class Goldeye(LanternCamera):
    def __init__(self):
        self.im = dao.shm('/tmp/testShm.im.shm', np.zeros((520, 656)).astype(np.uint16))
        self.ditshm = dao.shm('/tmp/testShmDit.im.shm', np.zeros((1,1)).astype(np.float32))
        self.ditshm.set_data(self.ditshm.get_data() * 0 + 220_000)
        self.gainshm = dao.shm('/tmp/testShmGain.im.shm', np.zeros((1,1)).astype(np.float32))
        self.gainshm.set_data(self.gainshm.get_data() * 0 + 10.0)
        self.fpsshm = dao.shm('/tmp/testShmFps.im.shm', np.zeros((1,1)).astype(np.float32))

    @property
    def exp_ms(self):
        return float(self.ditshm.get_data()[0][0] / 1000)

    @exp_ms.setter
    def exp_ms(self, val):
        self.set_exp_ms(val)
            
    def set_exp_ms(self, val, remind=True):
        # TODO don't overwrite previous darks
        assert val > 0, "invalid value for exp_ms"
        val = float(val)
        self.ditshm.set_data(self.ditshm.get_data() * 0 + val * 1000)
        dark_filepath = self.filepath(f"dark_exptime_ms_{val}_gain_{self.gain}")
        if path.isfile(dark_filepath):
            self.dark = np.load(dark_filepath)
        elif remind:
            print("New exposure time requires new dark frame, flat, and interaction matrix")
  
    @property
    def gain(self):
        return float(self.gainshm.get_data()[0][0])

    @gain.setter
    def gain(self, val):
        assert 0 <= val <= 18, "invalid value for gain"
        val = float(val)
        self.gainshm.set_data(self.gainshm.get_data() * 0 + val)
        dark_filepath = self.filepath(f"dark_exptime_ms_{self.exp_ms}_gain_{val}")
        if path.isfile(dark_filepath):
            self.dark = np.load(dark_filepath)
        else:
            print("New gain requires new dark frame, flat, and interaction matrix")
  
    @property
    def fps(self):
        return self.fps.get_data()

    @fps.setter
    def set_fps(self, val):
        self.fpsshm.set_data(self.fps.get_data() * 0 + val)
        
class SimLanternCamera(LanternCamera):
    def __init__(self):
        pass