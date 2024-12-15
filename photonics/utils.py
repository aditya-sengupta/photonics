from datetime import datetime
import numpy as np
import os
from os import path
from copy import copy
from hcipy import Field, imshow_field
from scipy.spatial import ConvexHull

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
DATA_PATH = path.join(PROJECT_ROOT, "data")
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
    
# following the ShaneAO convention
zernike_names = [
    "tip", "tilt", "focus", "astig", "astig45", "coma90", "coma", "tricoma90", "tricoma", "spherical", "astig5th45", "astig5th"
] + [f"Z{i}" for i in range(12, 82)]
    
def is_list_or_dim1_array(x):
    return isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1)

def date_now():
    return datetime.now().strftime("%Y%m%d")[2:]

def time_now():
    return datetime.now().strftime("%H%M")

def datetime_now():
    return datetime.now().isoformat().replace(":", ".")

def rms(x):
    return np.sqrt(np.mean((x - np.mean(x)) ** 2))

def rms_aperture(x, aperture):
    return rms(x[aperture != 0.0])

def datetime_ms_now():
    dt = datetime.now()
    return dt.strftime('%Y%m%d')[2:] + "_" + dt.strftime('%H%M%S') + "_" + str(dt.microsecond)

def time_ms_now():
    dt = datetime.now()
    return dt.strftime('%H%M%S') + "_" + str(dt.microsecond)

def make_fname(n):
    return f"{PROJECT_ROOT}/data/pl_{date_now()}/{n}_{datetime_now()}"

def angles_relative_to_center(x, y):
    xc, yc = np.mean(x), np.mean(y)
    xd, yd = x - xc, y - yc
    return (np.arctan2(yd, xd) + 3 * np.pi / 2) % (2 * np.pi)

def lmap(f, x):
    return list(map(f, x))

def nanify(phase_screen, aperture=None):
    if aperture is None:
        aperture = phase_screen
    x = copy(phase_screen)
    x = x - np.mean(x)
    x[np.where(aperture == 0)] = np.nan
    return Field(x, phase_screen.grid)

def imshow_psf(f: Field, **kwargs):
    imshow_field(np.log10(f / np.max(f)), **kwargs)

def peak_to_valley(x):
    return np.max(x) - np.min(x)

def norm(a, b):
    return np.sum(a * np.conj(b))

def corr(a, b):
    return np.abs(norm(a, b)) / np.sqrt(norm(a, a) * norm(b, b))

def center_of_mass(img, indices_x, indices_y):
    # Returns the center of mass (intensity-weighted sum) in the x and y direction of the image.
    xc = np.sum(img * indices_x) / np.sum(img)
    yc = np.sum(img * indices_y) / np.sum(img)
    return xc, yc

def ports_in_radial_order(points):
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

    return np.flip(xnew), np.flip(ynew)

def refine_centroids(centroids, image, cutout_size=12):
    new_centroids = np.zeros_like(centroids)
    for i in range(len(centroids)):
        xl, xu = int(centroids[i,0]) - cutout_size, int(centroids[i,0]) + cutout_size + 1
        yl, yu = int(centroids[i,1]) - cutout_size, int(centroids[i,1]) + cutout_size + 1
        new_centroids[i] = center_of_mass(image[xl:xu, yl:yu], xg[xl:xu, yl:yu], yg[xl:xu, yl:yu])
        
    return new_centroids

def normalize(x):
    return x / np.sum(x)