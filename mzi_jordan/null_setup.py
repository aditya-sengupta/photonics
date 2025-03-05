import numpy as np
import hcipy as hc

def ef_to_wf(electric_field, focal_grid, wavelength):
    '''Electric Field as numpy array to hcipy.Field
    
    Takes an electric field as a Numpy array (one dimensional array) and converts it into an HCIPy Field object,
    then it is converted into an HCIPy Wavefront object to be injected into an SMF/waveguide
    For this, we also need a focal grid corresponding to the lantern output port you are using
    Parameters 
    ----------
    electric_field: scalar or complex electric field
        EF of an output port from lantern 
    focal_grid: Grid
        Grid where the output port is defined
    wavelength: scalar 
        The wavelength of the wavefront/beam
    
    Returns
    -------
    wavefront
        The EF as an HCIPy Wavefront object
    '''
    field_temp = hc.Field(electric_field, focal_grid) #if your numpy array is two-dimensional add .ravel() after electric_field
    wavefront = hc.Wavefront(field_temp, wavelength = wavelength)
    return wavefront


def wf_to_smf(wavefront, core_radius, fiber_NA, fiber_length):
    '''Couple wavefront to SMF/waveguide
    
    Takes the output wavefront from a lantern port and couples it into an SMF/waveguide. Strictly speaking, we are not coupling the 
    output of the lantern since a real lantern can have its outputs as individual SMF, but it provides us with an HCIPy object that we can 
    handle more easily with other HCIPy functions (eg. phase, amplitude, total power)
    
    Parameters
    ----------
    wavefront: Wavefront
        EF as an HCIPy Wavefront object
    core_radius: scalar
        Core radius of the SMF/waveguide
    fiber_NA: scalar
        The numerical aperture of the SMF/waveguide, given by the core and cladding refractive indices of your photonic lantern, use:
        NA = sqrt(ncore**2 - nclad**2)
    fiber_length: scalar
        The length of fiber/waveguide through which the beam is propagated
    
    Returns
    -------
    smf_wf
        The EF of the beam after being propagated through the fiber length, as an HCIPy Wavefront object
    '''
    single_mode_fiber = hc.StepIndexFiber(core_radius, fiber_NA, fiber_length)
    smf_wf = single_mode_fiber.forward(wavefront)
    return smf_wf


def directional_coupler_sym(beam_1, beam_2):
    '''Symmetric Directional Coupler/beamsplitter with 50/50 splitting ratio
    
    Takes two wavefronts (EFs), combines them resulting in two outputs. The DC/beamsplitter can be expressed by a matrix
    M = [[cos(kL), i*sin(kL)], [i*sin(kL), cos(kL)]], 
    where kL is the product of the coupling constant and the coupling length of 
    the directional coupler; depending on the latter, we can adjust the splitting ratio. For our purposes, we consider an ideal symmetric
    coupler, that is 50/50 ==> M = [[sqrt(0.5), i*sqrt(0.5)], [i*sqrt(0.5), sqrt(0.5)]]
    Note: For the splitting of the beams to be 50/50, they inputs need to have the SAME PHASE
    
    Parameters
    ----------
    beam_1: complex electric field
        Input beam 1 as an HCIPy Wavefront object
    beam_2: complex electric field
        Input beam 2 as an HCIPy Wavefront object
    
    Returns
    -------
    out_beam_1
        EF of one of the beams resulting from the combination of two inputs. HCIPy Wavefront object
    out_beam_2
        EF of one of the beams resulting from the combination of two inputs. HCIPy Wavefront object
    '''
    out_beam_1 = hc.Wavefront(np.sqrt(0.5) * beam_1.electric_field + np.sqrt(0.5) * beam_2.electric_field * 1j, wavelength=beam_1.wavelength)
    out_beam_2 = hc.Wavefront(1j * np.sqrt(0.5) * beam_1.electric_field + np.sqrt(0.5) * beam_2.electric_field, wavelength=beam_1.wavelength)
    return out_beam_1, out_beam_2

def MZI_total(beam_1, beam_2, out=1, add_phase=None):
    '''Mach Zehnder Interferometer comprised of two directional couplers with a phase shifter before each DC that adjusts the
    phases to have constructive/destructive interference in the outputs
    
    Takes two wavefronts (EFs), and combines them through two DCs. The first phase shifter matches the phases of the inputs so the DC splits light
    into 50/50. The second phase shifter adjust the relative phase between the outputs of the first DC to have constructive/destructive interference,
    i.e. all light goes into one of the outputs of the second DC
    
    Parameters
    ----------
    beam_1: complex electric field
        Input beam 1 as an HCIPy Wavefront object
    beam_2: complex electric field
        Input beam 2 as an HCIPy Wavefront object
    out: output from which all light will come out (for total constructive/destructive interference)
        If out=1, constructive interference in output 1, and destructive interference in output 2
        If out=2, constructive interference in output 2, and destructive interference in output 1
    add_phase: extra phase to add to second phase shifter
        Default = 0 (none), so that constructive/destructive interference occurs. If different, splitting ratio of outputs will
        change
        
    Returns
    -------
    out_beam_1
        EF of one of the beams resulting from the combination of two inputs. HCIPy Wavefront object
    out_beam_2
        EF of one of the beams resulting from the combination of two inputs. HCIPy Wavefront object
    '''
    #1st phase shifter - phase matching
    diff_1 = beam_1.phase[0] - beam_2.phase[0] + 2*np.pi
    #1st DC - 50/50 splitting, matches phase ofbeam 2 to phase of beam 1
    beam_1_temp = hc.Wavefront(np.sqrt(0.5) * beam_1.electric_field + 1j * np.sqrt(0.5) * beam_2.electric_field * np.exp(1j * diff_1), wavelength=beam_1.wavelength)
    beam_2_temp = hc.Wavefront(1j * np.sqrt(0.5) * beam_1.electric_field + np.sqrt(0.5) * beam_2.electric_field * np.exp(1j * diff_1), wavelength=beam_1.wavelength)
    
    #2nd phase shifter - 
    if add_phase==None:
        add_phase = 0
    else:
        add_phase = add_phase
    if out==1:
        #adds an additional pi phase shift to have constructive interference in output 1
        diff_2 = beam_1_temp.phase[0] - beam_2_temp.phase[0] + 3*np.pi/2 + add_phase
    elif out==2:
        #phase difference needed for destructive interference in output 1
        diff_2 = beam_1_temp.phase[0] - beam_2_temp.phase[0] + np.pi/2 + add_phase
    #1st DC - 50/50 splitting, matches phase ofbeam 2 to phase of beam 1
    beam_1_out = hc.Wavefront(np.sqrt(0.5) * beam_1_temp.electric_field + 1j * np.sqrt(0.5) * beam_2_temp.electric_field * np.exp(1j * diff_2), wavelength=beam_1_temp.wavelength)
    beam_2_out = hc.Wavefront(1j * np.sqrt(0.5) * beam_1_temp.electric_field + np.sqrt(0.5) * beam_2_temp.electric_field * np.exp(1j * diff_2), wavelength=beam_1_temp.wavelength)
    return beam_1_out, beam_2_out
    
    