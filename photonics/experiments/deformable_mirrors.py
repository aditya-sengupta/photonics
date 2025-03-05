import time
import zmq
from abc import ABC
from tqdm import trange
import numpy as np
import hcipy as hc

class ShaneDM:
    def __init__(self):
        self.Nmodes = 12
        self.actuators = np.zeros(self.Nmodes)
        self.setup_paramiko()
        
    def setup_paramiko(self):
        host = "karnak.ucolick.org"
        username = "user"
        password = "yam != spud"

        self.client = paramiko.client.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(host, username=username, password=password)
        
    def __del__(self):
        self.client.close()
        
    def apply_mode(self, z, amp, verbose=True):
        assert isinstance(z, int), "first argument must be an integer (Zernike number)"
        assert isinstance(amp, float), "second argument must be a float (amplitude)"
        self.actuators[:] = 0.0
        self.actuators[z-1] = amp
        self.command_to_dm(verbose=verbose)
        
    def command_to_dm(self, p, verbose=True):
        """
        Send a command to the ShaneAO woofer.

        Parameters:
            amplitudes - list or np.ndarray
            The amplitude of each mode in [1, 2, ..., Nmodes], in order.
        """
        self.actuators[:] = p[:]
        assert len(self.actuators) == self.Nmodes, "wrong number of modes specified"
        assert np.all(np.abs(self.actuators) <= 1.0), "sending out-of-bounds amplitudes"
        command = ",".join(map(str, self.actuators))
        if verbose:
            print(f"DMC {command}.")
        #warnings.warn("If you see this and you're at Lick, uncomment the lines defining and running shell_command.")
        self.client.exec_command(f"/home/user/ShaneAO/shade/imageSharpen_nogui -s {command}")
        
    def send_zeros(self, verbose=True):
        self.actuators[:] = 0.0
        if verbose:
            print("Sending zeros.")
        self.command_to_dm(verbose=False)
        
class SimulatedDM:
    def __init__(self, pupil_grid, dm_basis="modal", num_actuators=9, telescope_diameter=10):
        self.Nmodes = num_actuators ** 2
        if dm_basis == "zonal":
            actuator_spacing = telescope_diameter / num_actuators
            influence_functions = hc.make_gaussian_influence_functions(pupil_grid, num_actuators, actuator_spacing)
            self.deformable_mirror = hc.DeformableMirror(influence_functions)
        elif dm_basis == "modal":
            modes = hc.make_zernike_basis(num_actuators ** 2, telescope_diameter, pupil_grid, starting_mode=2)
            self.deformable_mirror = hc.DeformableMirror(modes)
        else:
            raise NameError("DM basis needs to be zonal or modal")
        
    def send_zeros(self, verbose=True):
        if verbose:
            print("Sending zeros.")
        self.deformable_mirror.actuators[:] = 0.0
        
    @property
    def actuators(self):
        # just for convenience; actuators setter is command_to_dm
        # wait, actually, why don't I just...do that
        return self.deformable_mirror.actuators
        
    def command_to_dm(self, p, verbose=False):
        self.deformable_mirror.actuators[:] = p[:]
        
    def apply_mode(self, mode, amplitude):
        self.deformable_mirror.actuators[mode] = amplitude

    def apply_segment(self, segment, z, xgrad, ygrad):
        print("Need to ask Maria to implement this")
        pass
    
    def forward(self, wavefront):
        # to be augmented with Maria's segmented DM code
        return self.deformable_mirror.forward(wavefront)
    