import numpy as np
try:
    import imageio.v3 as iio
except ModuleNotFoundError:
    import imageio as iio

import os
from os import path
from astropy.stats import sigma_clipped_stats
from math import floor, ceil
from matplotlib import pyplot as plt
from photutils.detection import DAOStarFinder
from scipy.spatial import ConvexHull
import warnings

from .utils import date_now, datetime_now, angles_relative_to_center, PROJECT_ROOT, DATA_PATH

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
    
    def set_centroids(self, img, min_energy=10000, imax=50):
        # initial finding
        """
        imgc = img.copy()
        locs = []
        i = 0
        s = 20
        while len(self.guess_positions) < self.nports and i < imax:
            ind = np.unravel_index(np.argmax(imgc, axis=None), img.shape)
            if np.sum(imgc[ind[0]-s:ind[0]+s, ind[1]-s:ind[1]+s]) > min_energy:
                self.guess_positions.append(ind)
            imgc[ind[0]-s:ind[0]+s,ind[1]-s:ind[1]+s] = 0
            i += 1
        """
        mean, median, std = sigma_clipped_stats(img, sigma=10.0)
        daofind = DAOStarFinder(fwhm=self.fwhm, threshold=self.threshold*std) 
        # fwhm tuned to the port sizes
        # threshold is just large relative to background
        sources = daofind(img - median)
        xc, yc = (np.asarray(sources[k]) for k in ["xcentroid", "ycentroid"])
        if len(xc) != self.nports:
            warnings.warn(f"should have found {self.nports} ports but found {len(xc)}")
            self.save_intensities = False
        refined_locs = np.vstack((xc, yc)).T

        """# COM refinement
        refined_locs = []
        for (xc, yc) in self.guess_positions:
            img_cut, xi_cut, yi_cut = img[xc-s:xc+s,yc-s:yc+s], self.xi[xc-s:xc+s,yc-s:yc+s], self.yi[xc-s:xc+s,yc-s:yc+s]
            xcom = np.sum(xi_cut * img_cut) / np.sum(img_cut)
            ycom = np.sum(yi_cut * img_cut) / np.sum(img_cut)
            refined_locs.append([xcom, ycom])"""

        self.xc, self.yc, self.radial_shell = self.ports_in_radial_order(np.array(refined_locs))

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

    def ports_in_radial_order(self, points):
        xnew, ynew = np.zeros_like(points[:,0]), np.zeros_like(points[:,1])
        prev_idx = 0
        radial_shell = np.zeros(len(xnew))
        k = 0
        while len(points) > 0: # should run three times for a 19-port lantern
            if len(points) == 1:
                v = np.array([0])
            else:
                hull = ConvexHull(points)
                v = hull.vertices
            nhull = len(v)
            xtemp, ytemp = points[v][:,0], points[v][:,1]
            sortperm = np.argsort(angles_relative_to_center(xtemp, ytemp))[::-1]
            xnew[prev_idx:prev_idx+nhull] = xtemp[sortperm]
            ynew[prev_idx:prev_idx+nhull] = ytemp[sortperm]
            radial_shell[prev_idx:prev_idx+nhull] = k
            k += 1
            prev_idx += nhull
            points = np.delete(points, v, axis=0)

        return np.flip(xnew), np.flip(ynew), np.flip((k - 1 - radial_shell) / (k - 1))

    def bounding_box(self, pad=5):
        return (floor(min(self.xc)-pad), ceil(max(self.xc)+pad), floor(min(self.yc)-pad), ceil(max(self.yc)+pad))

    def crop_to_bounding_box(self, img, savename=None, pad=5):
        bbox = self.bounding_box(pad=pad)
        if len(img.shape) == 3:
            crop_img = img[:,bbox[2]:bbox[3], bbox[0]:bbox[1]]
        else:
            crop_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

        if savename is not None:
            np.save(self.filepath(savename, ext="npy"), crop_img)

        return crop_img

    def remove_port(self, idx):
        np.delete(self.xc, idx)
        np.delete(self.yc, idx)
        self.xc, self.yc, self.radial_shell = self.ports_in_radial_order(np.vstack((self.xc, self.yc)))

    def add_port(self, x, y):
        self.xc = np.append(self.xc, x)
        self.yc = np.append(self.yc, y)
        self.xc, self.yc, self.radial_shell = self.ports_in_radial_order(np.vstack((self.xc, self.yc)).T)

    def get_intensities(self, img, exclude=[]):
        mask = np.zeros_like(img)
        intensities = np.zeros(self.nports - len(exclude))
        j = 0
        for i in range(self.nports - len(exclude)):
            while j in exclude:
                j += 1
            intensities[i] = np.average(img[np.where((self.xi - self.xc[j]) ** 2 + (self.yi - self.yc[j]) ** 2 <= self.fwhm ** 2)])
            j += 1

        return intensities

    def port_mask(self):
        mask = np.zeros(self.imgshape)
        for (xv, yv) in zip(self.xc, self.yc):
            mask[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= self.fwhm ** 2)] = 1

        return mask

    def reconstruct_image(self, img, intensities):
        recon_image = np.zeros_like(img)
        for (xv, yv, intensity) in zip(self.xc, self.yc,intensities):
            recon_image[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= self.fwhm ** 2)] = intensity

        return recon_image

    def save(self, fname, data, ext="npy", verbose=True):
        fpath = self.filepath(fname, ext=ext)
        if verbose:
            print(f"Saving data to {fpath}")
        np.save(self.filepath(fname, ext=ext), data)

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
