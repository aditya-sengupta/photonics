import numpy as np
try:
    import imageio.v3 as iio
except ModuleNotFoundError:
    import imageio as iio

import os
from os import path
from astropy.stats import sigma_clipped_stats
from math import floor, ceil
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
from scipy.spatial import ConvexHull
import warnings

from ..utils import date_now, datetime_now, angles_relative_to_center, DATA_PATH

class LanternReader:
    """
    A LanternReader contains information about how to interpret a photonic lantern image. 

    During initialization, it identifies the number of ports in the image and their locations, and 
    """
    def __init__(self, nports, fwhm, threshold, ext, imgshape, guess_positions=[], subdir=None):
        self.nports = nports
        self.ext = ext
        self.fwhm = fwhm
        self.imgshape = imgshape
        self.yi, self.xi = np.indices(imgshape)
        self.guess_positions = guess_positions
        if subdir is None:
            subdir = f"pl_{date_now()}"
        self.subdir = subdir
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        print(f"Path for data saving set to {self.directory}")
        self.threshold = threshold
        self.saturation = 60_000
        self.save_intensities = True # if False, save full images

    def read_image(self, fname):
        path_to_file = self.filepath(fname)
        if self.ext == "raw":
            return np.fromfile(path_to_file, dtype="int16", sep="").reshape(self.imgshape)
        elif self.ext == "npy":
            return np.load(path_to_file)
        else:
            img = iio.imread(path_to_file)
            if len(img.shape) == 3:
                return img[:,:,0]
            return img

    def bounding_box(self, pad=0):
        return (floor(min(self.xc)-pad), ceil(max(self.xc)+pad), floor(min(self.yc)-pad), ceil(max(self.yc)+pad))

    def crop_to_bounding_box(self, img, savename=None, pad=15):
        bbox = self.bounding_box(pad=pad)
        if len(img.shape) == 3:
            crop_img = img[:,bbox[2]:bbox[3], bbox[0]:bbox[1]]
        else:
            crop_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

        if savename is not None:
            np.save(self.filepath(savename, ext="npy"), crop_img)

        return crop_img

    def saturation_map(self, img):
        plt.imshow(img >= self.saturation)
        plt.title("Saturation map")
        plt.show()

    def peaks_per_port(self, img):
        r = self.fwhm
        max_vals = []
        # TODO don't repeat the coordinate calculation each time
        for (x, y) in zip(self.xc, self.yc):
            lower_x, upper_x = int(np.floor(x - r)), int(np.ceil(x + r))
            lower_y, upper_y = int(np.floor(y - r)), int(np.ceil(y + r))
            masked = img[
                lower_y:upper_y,
                lower_x:upper_x
            ]
            max_vals.append(np.max(masked))
        
        return np.array(max_vals)

    def saturating_ports(self, img):
        return np.array(self.peaks_per_port(img)) > self.saturation

    def plot_saturating_ports(self, img):
        port_booleans = self.saturating_ports(img)
        return self.reconstruct_image(img, port_booleans)
