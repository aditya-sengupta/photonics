from datetime import datetime
import numpy as np
import os
from os import path

PROJECT_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
DATA_PATH = path.join(PROJECT_ROOT, "data")
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
    
def is_list_or_dim1_array(x):
    return isinstance(x, list) or (isinstance(x, np.ndarray) and len(x.shape) == 1)

def date_now():
    return datetime.now().strftime("%Y%m%d")[2:]

def time_now():
    return datetime.now().strftime("%H%M")

def datetime_now():
    return date_now() + "_" + time_now()

def rms(x):
    return np.sqrt(np.sum(x ** 2))

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
