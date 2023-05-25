import numpy as np
import imageio.v3 as iio

from astropy.stats import sigma_clipped_stats
from matplotlib import pyplot as plt
from os import path
from photutils.detection import DAOStarFinder

class LanternReader:
    def __init__(self, nports, fwhm, ext, imgshape):
        self.nports = nports
        self.fwhm = fwhm
        self.ext = ext
        self.imgshape = imgshape
        self.yi, self.xi = np.indices(imgshape)

    def read_image(self, p):
        return iio.imread(path.join("..", "data", p + "." + self.ext))[:,:,0]
    
    def set_centroids(self, img):
        mean, median, std = sigma_clipped_stats(img, sigma=3.0)
        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=20.*std) 
        # fwhm tuned to the port sizes
        # threshold is just large relative to backgrouhd
        sources = daofind(img - median)
        xc, yc = (np.asarray(sources[k]) for k in ["xcentroid", "ycentroid"])
        assert len(xc) == self.nports, f"should have found {nports} ports but found {len(xc)}"
        self.xc, self.yc = self.ports_in_radial_order(xc, yc)

    def ports_in_radial_order(self, xc, yc):
        # sort everything in radial order
        return xc, yc

    def get_intensities(self, img):
        mask = np.zeros_like(img)
        intensities = np.zeros(self.nports)
        for (i, (xv, yv)) in enumerate(zip(self.xc, self.yc)):
            intensities[i] = np.average(img[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= self.fwhm ** 2)])

        return intensities

    def reconstruct_image(self, img, intensities):
        recon_image = np.zeros_like(img)
        for (xv, yv, intensity) in zip(self.xc, self.yc,intensities):
            recon_image[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= self.fwhm ** 2)] = intensity

        return recon_image
