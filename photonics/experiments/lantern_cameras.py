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
    def __init__(self, reference_image, tag=None, rerun=False):
        self.reference_image = reference_image
        self.xi, self.yi = np.indices(self.reference_image.shape)
        self.spot_radius_px = 10
        if tag is None:
            self.tag = input("enter a tag: ")
        else:
            self.tag = tag
        self.set_centroids(rerun=rerun)

        
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
    
    def set_centroids(self, rerun=False):
        """
        Sets the centroids of each PL port based on a reference image
        """
        centroids_file = os.path.join(DATA_PATH, "current_centroids", f"current_centroids_{self.tag}.hdf5")
        if os.path.exists(centroids_file) and not rerun:
            with h5py.File(centroids_file, "r") as f:
                centroids = np.array(f["centroids"])
                self.xc = centroids[:,0]
                self.yc = centroids[:,1]
                self.radial_shell = centroids[:,2]
                self.spot_radius_px = f["centroids"].attrs["spot_radius_px"]
            self.Nports = len(self.xc)
            self.masks = [(self.xi - xc) ** 2 + (self.yi - yc) ** 2 <= self.spot_radius_px ** 2 for (xc, yc) in zip(self.xc, self.yc)]
            return
        global coords
        accepted_positions = False
        while not accepted_positions:
            while len(coords) > 0:
                coords.remove(coords[0])
            test_image_ret = self.reference_image
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

        self.xc, self.yc, self.radial_shell = ports_in_radial_order(self.refine_centroids(centroids))
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
                
    def refine_centroids(self, centroids, cutout_size=25):
        new_centroids = np.zeros_like(centroids)
        self.Nports = centroids.shape[0]
        for i in range(self.Nports):
            xl, xu = int(centroids[i,0]) - cutout_size, int(centroids[i,0]) + cutout_size + 1
            yl, yu = int(centroids[i,1]) - cutout_size, int(centroids[i,1]) + cutout_size + 1
            image_cutout = self.reference_image[yl:yu, xl:xu]
            new_centroids[i] = center_of_mass(image_cutout - np.min(image_cutout) + 1, self.xi[yl:yu, xl:xu], self.yi[yl:yu, xl:xu])
            
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
        