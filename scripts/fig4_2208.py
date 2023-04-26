from fig4_2208_config import *
import sys, subprocess

if __name__ == "__main__":
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')
    zern = int(sys.argv[1])
    if len(sys.argv) > 2:
        ampls = [float(sys.argv[2])]
    else:
        ampls = np.linspace(-1, 1, 21)
    
    for ampl in ampls:
        save_for_zampl(zern, ampl)

