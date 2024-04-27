import numpy as np
import hcipy as hc
import os
from matplotlib import pyplot as plt
from tqdm import trange
from .utils import PROJECT_ROOT, date_now

class PyramidOptics:
    def __init__(self, opt):
        """
        Make a pyramid wavefront sensor based on an optical setup (SecondStageOptics).
        """
        self.wl = opt.wl
        self.setup(opt)
        self.make_command_matrix(opt)

    def setup(self, opt):
        # set up pyramid wavefront sensor
        self.wl = opt.wl
        pixels_pyramid_pupils = 20 # number of pixels across each pupil; want 120 %(mod) pixels_pyramid_pupils =0. VARY THIS PARAMETER
        pwfs_grid = hc.make_pupil_grid(pixels_pyramid_pupils*2,opt.telescope_diameter*2)

        mld = 5
        modradius = mld * opt.wl / opt.telescope_diameter # modulation radius in radians;
        self.modsteps = 4 # needs to be a factor of 4

        pwfs = hc.PyramidWavefrontSensorOptics(opt.pupil_grid, pwfs_grid, separation=opt.telescope_diameter, wavelength_0=opt.wl)
        self.mpwfs = hc.ModulatedPyramidWavefrontSensorOptics(pwfs,modradius,self.modsteps)
        self.wfs_camera = hc.NoiselessDetector(pwfs_grid)
        self.image_ref = self.readout(opt.wf)
        
    def readout(self, wf):
        wf_mpwfs = self.mpwfs(wf)
        for m in range(self.modsteps):
            self.wfs_camera.integrate(wf_mpwfs[m], 1)

        img = self.wfs_camera.read_out()
        img /= img.sum()
        return img
        
    def make_command_matrix(self, opt, rerun=False):
        cmd_path = PROJECT_ROOT + f"/data/secondstage_pyramid/cm_{date_now()}.npy"
        if (not rerun) and os.path.exists(cmd_path):
            self.command_matrix = np.load(cmd_path)
        else:
            probe_amp = 0.01 * self.wl
            num_modes = opt.deformable_mirror.num_actuators
            slopes = []

            for ind in trange(num_modes):
                slope = 0

                # Probe the phase response
                for s in [1, -1]:
                    amp = np.zeros((num_modes,))
                    amp[ind] = s * probe_amp
                    opt.deformable_mirror.actuators = amp
                    dm_wf = opt.deformable_mirror.forward(opt.wf)
                    image = self.readout(dm_wf)
                    slope += s * (image-self.image_ref)/(2 * probe_amp)

                slopes.append(slope)

            slopes = hc.ModeBasis(slopes)
            self.command_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=1e-3, svd=None)
            np.save(PROJECT_ROOT + f"/data/secondstage_pyramid/cm_{date_now()}.npy", self.command_matrix)
    
