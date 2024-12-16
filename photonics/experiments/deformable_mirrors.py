import time
import zmq
from tqdm import trange
import numpy as np
import hcipy as hc

class ShaneDM:
    def __init__(self):
        self.curr_dmc = np.zeros(Nmodes)
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
        self.curr_dmc[:] = 0.0
        self.curr_dmc[z-1] = amp
        self.command_to_dm(verbose=verbose)
        
    def command_to_dm(self, verbose=True):
        """
        Send a command to the ShaneAO woofer.

        Parameters:
            amplitudes - list or np.ndarray
            The amplitude of each mode in [1, 2, ..., Nmodes], in order.
        """
        assert len(self.curr_dmc) == self.Nmodes, "wrong number of modes specified"
        assert np.all(np.abs(self.curr_dmc) <= 1.0), "sending out-of-bounds amplitudes"
        command = ",".join(map(str, self.curr_dmc))
        if verbose:
            print(f"DMC {command}.")
        #warnings.warn("If you see this and you're at Lick, uncomment the lines defining and running shell_command.")
        self.client.exec_command(f"/home/user/ShaneAO/shade/imageSharpen_nogui -s {command}")
        
    def send_zeros(self, verbose=True):
        self.curr_dmc[:] = 0.0
        if verbose:
            print("Sending zeros.")
        self.command_to_dm(verbose=False)
        
class IrisDM:
    def __init__(self):
        port = "5557"
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://128.114.23.114:%s" % port)
        self.socket.send(np.array([1]).astype(np.float32));
        msg = socket.recv()
        
    def apply_segment(self, segment, z, xgrad, ygrad):
        self.socket.send(np.array([segment, z, xgrad, ygrad]).astype(np.float32));
        return self.socket.recv()

    def apply_mode(self, mode, amplitude):
        self.socket.send(np.array([mode, amplitude]).astype(np.float32));
        return self.socket.recv()

    def release_mirror(self):
        self.socket.send(np.array([0]).astype(np.float32));
        msg = self.socket.recv()

    def stop_server(self):
        self.socket.send(np.array([2]).astype(np.float32));
        return self.socket.recv()

    def piston_all(self, pauseTime=0.1):
        # piston each segment
        for k in trange(1, 168):
            self.apply_segment(k, 0.2, 0, 0)
            time.sleep(pauseTime)
            self.apply_segment(k, -0.2, 0, 0)
            time.sleep(pauseTime)
            self.apply_segment(k, 0., 0, 0)
            
    def __del__(self):
        self.release_mirror()
        self.stop_server()
          
class SimulatedDM:
    def __init__(self, pupil_grid, dm_basis="modal", num_actuators=9, telescope_diameter=10):
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
        
    def apply_mode(self, mode, amplitude):
        self.deformable_mirror.actuators[mode] = amplitude

    def apply_segment(self, segment, z, xgrad, ygrad):
        print("Need to ask Maria to implement this")
        pass
    
    def forward(self, wavefront):
        # to be augmented with Maria's segmented DM code
        return self.deformable_mirror.forward(wavefront)
    