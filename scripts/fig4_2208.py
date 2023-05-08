from fig4_2208_config import *
import sys, subprocess
from itertools import product

if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')
    if len(sys.argv) > 1:
        zern = int(sys.argv[1])
    else:
        zerns = [2, 3, 4, 5, 6]

    if len(sys.argv) > 2:
        ampls = [float(sys.argv[2])]
    else:
        ampls = np.linspace(-1, 1, 11)
    
    for (zern, ampl) in product(zerns, ampls):
        save_for_zampl(zern, ampl)
