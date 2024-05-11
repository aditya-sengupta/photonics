import hcipy as hc
from .command_matrix import make_command_matrix

class PyramidOptics:
    def __init__(self, optics, pixels_pyramid_pupils=20, mld=5, modsteps=12):
        """
        Make a pyramid wavefront sensor based on an optical setup.
        """
        self.name = "pyramid"
        pwfs_grid = hc.make_pupil_grid(pixels_pyramid_pupils*2,optics.telescope_diameter*2)
        modradius = mld * optics.wl / optics.telescope_diameter
        self.modsteps = modsteps
        assert modsteps % 4 == 0
        pwfs = hc.PyramidWavefrontSensorOptics(optics.pupil_grid, pwfs_grid, separation=optics.telescope_diameter, wavelength_0=optics.wl)
        self.mpwfs = hc.ModulatedPyramidWavefrontSensorOptics(pwfs,modradius,self.modsteps)
        self.wfs_camera = hc.NoiselessDetector(pwfs_grid)
        make_command_matrix(optics, self, optics.wf, dm_basis=optics.dm_basis)

    def readout(self, wf):
        wf_mpwfs = self.mpwfs(wf)
        for m in range(self.modsteps):
            self.wfs_camera.integrate(wf_mpwfs[m], 1)

        img = self.wfs_camera.read_out()
        img /= img.sum()
        return img
    
    def reconstruct(self, wf: hc.Wavefront):
        """
        Takes in a wavefront and returns its equivalent DM command as reconstructed on the pyramid.
        
        Parameters
        ----------
        wf : hcipy.Wavefront
            The input aberrated wavefront.
            
        Returns
        -------
        recon : np.ndarray
            The reconstructed DM command.
        """
        img = self.readout(wf)
        recon = self.command_matrix.dot(img - self.image_ref)
        return recon
    