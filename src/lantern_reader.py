import numpy as np
import imageio.v3 as iio

from astropy.stats import sigma_clipped_stats
from matplotlib import pyplot as plt
from os import path
from photutils.detection import DAOStarFinder
from scipy.spatial import ConvexHull

def angles_relative_to_center(x, y):
    xc, yc = np.mean(x), np.mean(y)
    xd, yd = x - xc, y - yc
    return (np.arctan2(yd, xd) + np.pi / 2) % (2 * np.pi)

class LanternReader:
    def __init__(self, nports, cutout_size, fwhm, ext, imgshape, guess_positions=[]):
        self.nports = nports
        self.cutout_size = cutout_size
        self.ext = ext
        self.fwhm = fwhm
        self.imgshape = imgshape
        self.yi, self.xi = np.indices(imgshape)
        self.guess_positions = guess_positions

    def read_image(self, p):
        return iio.imread(path.join("..", "data", p + "." + self.ext))[:,:,0]
    
    def set_centroids(self, img, min_energy=10000):
        # initial finding
        imgc = img.copy()
        locs = []
        i = 0
        s = self.cutout_size
        while len(self.guess_positions) < self.nports and not(np.all(imgc == 0)):
            ind = np.unravel_index(np.argmax(imgc, axis=None), img.shape)
            if np.sum(imgc[ind[0]-s:ind[0]+s, ind[1]-s:ind[1]+s]) > min_energy:
                self.guess_positions.append(ind)
                imgc[ind[0]-s:ind[0]+s,ind[1]-s:ind[1]+s] = 0
        
        # COM refinement
        refined_locs = []
        for (xc, yc) in self.guess_positions:
            img_cut, xi_cut, yi_cut = img[xc-s:xc+s,yc-s:yc+s], self.xi[xc-s:xc+s,yc-s:yc+s], self.yi[xc-s:xc+s,yc-s:yc+s]
            xcom = np.sum(xi_cut * img_cut) / np.sum(img_cut)
            ycom = np.sum(yi_cut * img_cut) / np.sum(img_cut)
            refined_locs.append([xcom, ycom])

        self.xc, self.yc = self.ports_in_radial_order(np.array(refined_locs))

    def ports_in_radial_order(self, points):
        xnew, ynew = np.zeros_like(points[:,0]), np.zeros_like(points[:,1])
        prev_idx = 0
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
            prev_idx += nhull
            points = np.delete(points, v, axis=0)

        return np.flip(xnew), np.flip(ynew)

    def get_intensities(self, img, fwhm=15):
        mask = np.zeros_like(img)
        intensities = np.zeros(self.nports)
        for (i, (xv, yv)) in enumerate(zip(self.xc, self.yc)):
            intensities[i] = np.average(img[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= fwhm ** 2)])

        return intensities

    def reconstruct_image(self, img, intensities, fwhm=15):
        recon_image = np.zeros_like(img)
        for (xv, yv, intensity) in zip(self.xc, self.yc,intensities):
            recon_image[np.where((self.xi - xv) ** 2 + (self.yi - yv) ** 2 <= fwhm ** 2)] = intensity

        return recon_image
