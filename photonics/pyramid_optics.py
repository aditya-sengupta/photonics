import numpy as np
import hcipy as hc
from tqdm import trange

class PyramidOptics:
    def __init__(self):
        self.optics_setup()
        self.turbulence_setup()
        self.dm_setup()
        self.pyramid_setup()
        self.make_pyramid_command_matrix()
    
    def optics_setup(self):
        grid_size = 120 #define number of pixels across our telescope aperture.
        self.D = 1  #define the telescope diameter in meters.
        self.pupil_grid = hc.make_pupil_grid(grid_size, diameter=self.D)  #define our aperture grid (pupil grid)
        telescope_aperture = hc.make_circular_aperture(self.D)  #this is a function that returns a telescope aperture. Note this is a function.
        self.telescope_pupil = telescope_aperture(self.pupil_grid)   #telescope aperture (primary mirror)

        #pick our wavelength to use for the simulation
        self.wl = 1.55e-6
        self.k = 2*np.pi/self.wl #wavenumber. Converts between microns and radians
        
        self.focal_grid = hc.make_focal_grid(q=4, num_airy=20,spatial_resolution=self.wl/self.D) # how we want to sample the grid that our psf will be on...think of this like our camera
        self.focal_propagator = hc.FraunhoferPropagator(self.pupil_grid, self.focal_grid)  #this encodes our fourier transform as it propagates things from the telescope to our focus.
        self.wf = hc.Wavefront(self.telescope_pupil,wavelength=self.wl) #electric field in hcipy
        self.wf.total_power = 1

    def turbulence_setup(self):
        # let's pick the properties of our turblence
        fried_parameter = 0.13 # meters
        outer_scale = 50 #  meters
        velocity = 10. # wind speed in m/s.
        Cn_squared = hc.Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2

        # make our atmospheric turbulence layer
        self.layer = hc.InfiniteAtmosphericLayer(self.pupil_grid, Cn_squared, outer_scale, velocity)
        
    def dm_setup(self):
        #make the DM
        num_actuators = 9
        actuator_spacing = self.D / num_actuators
        influence_functions = hc.make_gaussian_influence_functions(self.pupil_grid, num_actuators, actuator_spacing)
        self.deformable_mirror = hc.DeformableMirror(influence_functions)

    def pyramid_setup(self):
        # set up pyramid wavefront sensor
        pixels_pyramid_pupils = 20 # number of pixels across each pupil; want 120 %(mod) pixels_pyramid_pupils =0. VARY THIS PARAMETER
        pwfs_grid = hc.make_pupil_grid(pixels_pyramid_pupils*2, self.D*2)

        mld = 5 # modulation radius in lambda/D (we will discuss modulation in a future lecture. Do not vary.)
        modradius = mld * self.wl / self.D # modulation radius in radians;
        self.modsteps = 4 # needs to be a factor of 4

        pwfs = hc.PyramidWavefrontSensorOptics(self.pupil_grid, pwfs_grid, separation=self.D, wavelength_0=self.wl)
        self.mpwfs = hc.ModulatedPyramidWavefrontSensorOptics(pwfs,modradius,self.modsteps)
        self.wfs_camera = hc.NoiselessDetector(pwfs_grid)
        
        self.image_ref = self.pyramid_readout(self.wf)
        
    def pyramid_readout(self, wf):
        for m in range(self.modsteps):
            self.wfs_camera.integrate(self.mpwfs(wf)[m], 1)

        img = self.wfs_camera.read_out()
        img /= img.sum()
        return img
        
    def make_pyramid_command_matrix(self):
        probe_amp = 0.01 * self.wl
        num_modes = self.deformable_mirror.num_actuators
        slopes = []

        for ind in trange(num_modes):
            slope = 0

            # Probe the phase response
            for s in [1, -1]:
                amp = np.zeros((num_modes,))
                amp[ind] = s * probe_amp
                self.deformable_mirror.actuators = amp
                dm_wf = self.deformable_mirror.forward(self.wf)
                image = self.pyramid_readout(dm_wf)
                slope += s * (image-self.image_ref)/(2 * probe_amp)

            slopes.append(slope)

        slopes = hc.ModeBasis(slopes)
        self.pyramid_command_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=1e-3, svd=None)
    
    