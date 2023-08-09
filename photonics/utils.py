from datetime import datetime
import numpy as np

date_now = lambda: datetime.now().strftime('%Y%m%d')[2:]
time_now = lambda: datetime.now().strftime('%H%M')
datetime_now = lambda: date_now() + "_" + time_now()

def datetime_ms_now():
    dt = datetime.now()
    return dt.strftime('%Y%m%d')[2:] + "_" + dt.strftime('%H%M%S') + "_" + str(dt.microsecond)

def time_ms_now():
    dt = datetime.now()
    return dt.strftime('%H%M%S') + "_" + str(dt.microsecond)

make_fname = lambda n: f"{PROJECT_ROOT}/data/pl_{date_now()}/{n}_{datetime_now()}"

def angles_relative_to_center(x, y):
    xc, yc = np.mean(x), np.mean(y)
    xd, yd = x - xc, y - yc
    return (np.arctan2(yd, xd) + 3 * np.pi / 2) % (2 * np.pi)
