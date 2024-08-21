from datetime import datetime
import numpy as np
import os
from os import path
from copy import copy
from hcipy import Field, imshow_field

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
    return datetime.now().isoformat()

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
