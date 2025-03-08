import os

from abc import ABC

import h5py
import numpy as np

from matplotlib import pyplot as plt
from hcipy import NoisyDetector, Wavefront, Field

from ..utils import onclick, coords, ports_in_radial_order, center_of_mass, datetime_now, normalize, DATA_PATH
from ..simulations.optics import Optics
from ..simulations.lantern_optics import LanternOptics

class LanternReader:
    def __init__(self, camera):
        self.camera = camera
    
    def measure_dark(self, direct=True):
        if direct:
            input("Taking a dark frame, remove the light source!")

        self.dark = 0
        self.dark = self.camera.get_image()
        # need to think through an architecture for saving, reusing, and not overwriting darks

    def measure_pl_flat(self):
        self.send_zeros(verbose=True)
        self.pl_flat = self.get_intensities(self.camera.get_image())
        self.save(f"pl_flat_{datetime_now()}", self.pl_flat)
        
    def plot_ports(self, save=False):
        sc = plt.scatter(self.xc, self.yc, c=self.radial_shell)
        plt.xlim((0, self.xg.shape[1]))
        plt.ylim((0, self.xg.shape[0]))
        plt.xticks([])
        plt.yticks([])
        sc.axes.invert_yaxis()
        for (i, (xc, yc)) in enumerate(zip(self.xc, self.yc)):
            plt.annotate(i + 1, (xc, yc), xytext=(xc+5, yc+5))

        if save:
            plt.savefig(self.filepath(f"port_mask_{datetime_now()}", ext="png"))
        plt.show()
    
    def set_centroids(self, image, rerun=False):
        """
        Sets the centroids of each PL port based on a reference image, in `image`.
        """
        if not hasattr(self, "tag"):
            self.tag = input("enter a tag: ")
        centroids_file = os.path.join(DATA_PATH, "current_centroids", f"current_centroids_{self.tag}.hdf5")
        if os.path.exists(centroids_file) and not rerun:
            with h5py.File(centroids_file, "r") as f:
                centroids = np.array(f["centroids"])
                self.xc = centroids[:,0]
                self.yc = centroids[:,1]
                self.radial_shell = centroids[:,2]
                self.spot_radius_px = f["centroids"].attrs["spot_radius_px"]
            return
        global coords
        while len(coords) > 0:
            coords.remove(coords[0])
        accepted_positions = False
        while not accepted_positions:
            test_image_ret = image
            fig = plt.figure()
            plt.imshow(test_image_ret)
            plt.title("Click on all the PL ports!")
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            centroids = np.array(coords)
            plt.imshow(test_image_ret, label="True coordinates")
            plt.scatter(*centroids.T, label="Clicked coordinates")
            plt.legend()
            plt.show()
            
            accepted_positions = input("Identified PL ports OK? [y/n] ") == "y"

        self.xc, self.yc, self.radial_shell = ports_in_radial_order(self.refine_centroids(centroids, image))
        self.yi, self.xi = np.indices(image.shape)
        self.masks = [(self.xi - xc) ** 2 + (self.yi - yc) ** 2 <= self.spot_radius_px ** 2 for (xc, yc) in zip(self.xc, self.yc)]
        self.centroids_dt = datetime_now()
        accepted_radius = False
        while not accepted_radius:
            fig, axs = plt.subplots(3,1, figsize=(3,9))
            for ax in axs:
                ax.set_axis_off()
            axs[0].imshow(test_image_ret)
            axs[1].imshow(sum(test_image_ret * m for m in self.masks))
            axs[2].imshow(self.reconstruct_image(test_image_ret, self.get_intensities(test_image_ret)))
            plt.show()
            
            update_radius = input("Reconstructed PL image OK? [y/new radius] ")
            if update_radius != "y":
                self.spot_radius_px = float(update_radius)
                self.masks = [(self.xi - xc) ** 2 + (self.yi - yc) ** 2 <= self.spot_radius_px ** 2 for (xc, yc) in zip(self.xc, self.yc)]
            else:
                accepted_radius = True
        new_centroids = np.zeros((len(centroids), 3))
        new_centroids[:,0] = self.xc
        new_centroids[:,1] = self.yc
        new_centroids[:,2] = self.radial_shell
        
        with h5py.File(centroids_file, "w") as f:
            c = f.create_dataset("centroids", data=new_centroids)
            c.attrs["spot_radius_px"] = self.spot_radius_px
                
    def refine_centroids(self, centroids, image, cutout_size=12):
        new_centroids = np.zeros_like(centroids)
        self.Nports = centroids.shape[0]
        for i in range(self.Nports):
            xl, xu = int(centroids[i,0]) - cutout_size, int(centroids[i,0]) + cutout_size + 1
            yl, yu = int(centroids[i,1]) - cutout_size, int(centroids[i,1]) + cutout_size + 1
            image_cutout = image[xl:xu, yl:yu]
            new_centroids[i] = center_of_mass(image_cutout - np.min(image_cutout) + 1, self.xg[xl:xu, yl:yu], self.yg[xl:xu, yl:yu])
            
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

class ShaneGoldeye:
    # for Shane; on muirSEAL get this from `seal`
    def __init__(self, dm): 
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
    
    def set_exp_ms(self, val):
        assert val > 0, "invalid value for exp_ms"
        val = float(val)
        self.ditshm.set_data(self.ditshm.get_data() * 0 + val * 1000)
        print("New exposure time requires new dark frame, flat, and interaction matrix")
  
    @property
    def gain(self):
        return float(self.gainshm.get_data()[0][0])

    @gain.setter
    def gain(self, val):
        assert 0 <= val <= 18, "invalid value for gain"
        val = float(val)
        self.gainshm.set_data(self.gainshm.get_data() * 0 + val)
        print("New gain requires new dark frame, flat, and interaction matrix")
  
    @property
    def fps(self):
        return self.fpsshm.get_data()

    @fps.setter
    def set_fps(self, val):
        self.fpsshm.set_data(self.fps.get_data() * 0 + val)
        
    def get_image(self):
        """
        Get an image off the lantern camera. 
        """
        frames = []
        for i in range(self.nframes):
            sleep(np.float64(self.exp_ms) / 1e3)
            frames.append(self.im.get_data(check=True).astype(float) - self.dark)

        return np.mean(frames, axis=0)
        
class SimulatedLanternCamera:
    def __init__(self, dm, optics, tag=""):
        self.tag = tag
        self.optics = optics
        self.lantern_optics = LanternOptics(optics)
        self.exp_ms = 1e5
        self.gain = 1
        self.nframes = 10
        self.detector = NoisyDetector(detector_grid=self.lantern_optics.focal_grid)
        self.dm = dm
        self.xg, self.yg = np.indices(optics.focal_grid.shape)
        self.spot_radius_px = 6
        
    def measure_dark(self):
        self.dark = Field(np.zeros(self.lantern_optics.focal_grid.size), grid=self.lantern_optics.focal_grid)
        # I could probably simulate this better
        
    def get_image(self):
        frames = []
        for _ in range(self.nframes):
            focal_wf = self.optics.focal_propagator(self.dm.forward(self.optics.wf))
            coeffs_true, pl_image = self.lantern_optics.lantern_output(focal_wf)
            pl_wf = Wavefront(Field(pl_image.flatten(), self.lantern_optics.focal_grid), wavelength=self.optics.wl)
            self.detector.integrate(pl_wf, self.exp_ms)
            frames.append(self.detector.read_out() - self.dark)
            
        return (sum(frames) / self.nframes).shaped