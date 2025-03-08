import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
from matplotlib import pyplot as plt

plt.rc('font', family='serif',size=12)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif" : "cmr10",
    "axes.formatter.use_mathtext" : True
})

from .utils import *
from .experiments.deformable_mirrors import SimulatedDM, ShaneDM
from .experiments.lantern_cameras import LanternReader, ShaneGoldeye
from .experiments.experiments import Experiments