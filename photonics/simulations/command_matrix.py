import hcipy as hc
import os
import numpy as np
from os.path import join
from ..utils import PROJECT_ROOT, date_now

def make_command_matrix(
    dm: hc.DeformableMirror, 
    wfs, # eventually this should be an hcipy.WavefrontSensor
    flat_wavefront: hc.Wavefront,
    probe_amp=1e-10,
    dm_basis="modal",
    rerun=False
):
    """
    Make a command matrix for a DM-WFS pair and assigns it to wfs.command_matrix
    """
    cmd_path = join(PROJECT_ROOT, "data", f"secondstage_{wfs.name}", f"cm_{date_now()}_{dm_basis}.npy")
    wfs.image_ref = wfs.readout(flat_wavefront)
    if (not rerun) and os.path.exists(cmd_path):
        wfs.command_matrix = np.load(cmd_path)
    else:
        num_modes = dm.num_actuators
        if hasattr(wfs, "nmodes"):
            num_modes = min(num_modes, wfs.nmodes)
        slopes = []
        for ind in range(num_modes):
            slope = 0
            for s in [1, -1]:
                amp = np.zeros((dm.num_actuators,))
                amp[ind] = s * probe_amp
                dm.actuators = amp
                dm_wf = dm.forward(flat_wavefront)
                image = wfs.readout(dm_wf)
                slope += s * (image - wfs.image_ref)/(2 * probe_amp)

            slopes.append(slope)

        slopes = hc.ModeBasis(slopes)
        wfs.interaction_matrix = slopes.transformation_matrix
        wfs.command_matrix = hc.inverse_tikhonov(slopes.transformation_matrix, rcond=1e-3, svd=None)
        np.save(cmd_path, wfs.command_matrix)