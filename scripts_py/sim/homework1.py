# %%
import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
from tqdm import trange

grid_size=120 #define number of pixels across our telescope aperture.
D=1  #define the telescope diameter in meters.
pupil_grid = make_pupil_grid(grid_size, diameter=D)  #define our aperture grid (pupil grid)
telescope_aperture  = make_circular_aperture(D)  #this is a function that returns a telescope aperture. Note this is a function.
telescope_pupil=telescope_aperture(pupil_grid)   #telescope aperture (primary mirror)

#pick our wavelength to use for the simulation
wavelength=1.55e-6
k=2*np.pi/wavelength #wavenumber. Converts between microns and radians

wf= Wavefront(telescope_pupil,wavelength=wavelength) #electric field in hcipy
wf.total_power = 1

focal_grid = make_focal_grid(q=4, num_airy=20,spatial_resolution=wavelength/D) # how we want to sample the grid that our psf will be on...think of this like our camera
propagator = FraunhoferPropagator(pupil_grid, focal_grid)  #this encodes our fourier transform as it propagates things from the telescope to our focus.

#reference image and the max for plotting the psf later as well as strehl ratio calculation
im_ref= propagator.forward(wf)
norm= np.max(im_ref.intensity)

# let's pick the properties of our turblence
fried_parameter = 0.1  # meters; VARY THIS PARAMETER from 0.05 (really bad) to 0.2 (really good)
outer_scale = 50 #  meters (this parameter does not really matter for our purposes today)
velocity = 10.0
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter)  #convert the fried parameter into Cn2

# make our atmospheric turbulence layer
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity, seed=1)

#make the DM
num_actuators = 9 # the number of actuators. VARY THIS PARAMETER
actuator_spacing = D / num_actuators
influence_functions = make_gaussian_influence_functions(pupil_grid, num_actuators, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)

# set up pyramid wavefront sensor
pixels_pyramid_pupils=20 # number of pixels across each pupil; want 120 %(mod) pixels_pyramid_pupils =0. VARY THIS PARAMETER
pwfs_grid = make_pupil_grid(pixels_pyramid_pupils*2, D*2)

mld=5 # modulation radius in lambda/D (we will discuss modulation in a future lecture. Do not vary.)
modradius = mld*wavelength/D # modulation radius in radians;
modsteps = 12 # keep this as a factor of 4. Increasing this value significantly increases the computation time. No need to change today.

pwfs = PyramidWavefrontSensorOptics(pupil_grid, pwfs_grid, separation=D, wavelength_0=wavelength)
mpwfs = ModulatedPyramidWavefrontSensorOptics(pwfs,modradius,modsteps)
wfs_camera = NoiselessDetector(pwfs_grid)

# #commands to modulate the PyWFS, get an image out, and calculate a reference
for m in range (modsteps) :
      wfs_camera.integrate(mpwfs(wf)[m], 1)

image_ref = wfs_camera.read_out()
image_ref /= image_ref.sum()

CM = np.load("../../data/secondstage_pyramid/cm_240510_zonal.npy")

#leaky integrator parameters (we will vary these in a future assignment)
gain = 0.3
leakage = 0.999

#AO loop speed: VARY THIS PARAMETER
dt=1./800

num_iterations = 200 #number of time steps in our simulation. We'll run for a total of dt*num_iterations seconds
sr=[] # so we can find the average strehl ratio
wavefronts_after_dm = []

layer.reset()
layer.t = 0
for timestep in trange(num_iterations):
    #get a clean wavefront
    wf_in=wf.copy()

    #evolve the atmospheric turbulence
    layer.t = timestep*dt

    #pass the wavefront through the turbulence
    wf_after_atmos = layer.forward((wf_in))

    #pass the wavefront through the DM for correction
    wf_after_dm = deformable_mirror.forward(wf_after_atmos)

    #send the wavefront containing the residual wavefront error to the PyWFS and get slopes
    wfs_wf = mpwfs.forward(wf_after_dm)
    for mmm in range (modsteps) :
              wfs_camera.integrate(wfs_wf[mmm], dt/modsteps)
    wfs_image = wfs_camera.read_out()
    wfs_image /= np.sum(wfs_image)
    diff_image = wfs_image - image_ref

    #Leaky integrator to calculate new DM commands
    deformable_mirror.actuators =  leakage*deformable_mirror.actuators - gain * CM.dot(diff_image)
    
    # Propagate to focal plane
    wf_focal = propagator.forward(wf_after_dm )

    wavefronts_after_dm.append(wf_after_dm)
    #calculate the strehl ratio to use as a metric for how well the AO system is performing.
    strehl_foc=get_strehl_from_focal(wf_focal.intensity/norm,im_ref.intensity/norm)
    sr.append(strehl_foc)

plt.plot(sr)
# %%
