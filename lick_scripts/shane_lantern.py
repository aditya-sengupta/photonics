# %%
import subprocess

from astropy.io import fits
from datetime import datetime
from functools import reduce
from time import sleep
from tqdm import tqdm, trange

Nmodes = 12 # for now

def save_telemetry(wait=0):
    sleep(wait)
    subprocess.run([
        "ssh", "-Y", 
        "user@shimmy.ucolick.org", "/opt/kroot/bin/modify",
        "-s", "saocon", "savedata=1"
    ])

def zern_to_dm(z, amp):
    amplitudes = [0] * 12
    amplitudes[z-1] = amp
    command_to_dm(amplitudes)

def command_to_dm(amplitudes):
    """
    Send a command to the ShaneAO woofer.

    Parameters:
        amplitudes - list or np.ndarray
        The amplitude of each mode in [1, 2, ..., Nmodes], in order.
    """
    assert len(amplitudes) == Nmodes, "wrong number of modes specified"
    assert np.all(np.abs(amplitudes) <= 5.0), "sending out-of-bounds amplitudes"
    command = ",".join(map(str, amplitudes))
    shell_command = ["ssh", "-Y", "gavel@shade.ucolick.org", "local/bin/imageSharpen", "-s", command]
    # print(" ".join(shell_command))
    subprocess.run(shell_command)

def send_zeros():
    command_to_dm([0] * 12)

def sweep_mode(z, min_amp=-1.0, max_amp=1.0, step=0.1, prompt=False):
    amplitudes = np.zeros(Nmodes)
    for a in tqdm(np.arange(min_amp, max_amp+step, step)):
        if prompt:
            print(a)
        amplitudes[z-1] = a
        command_to_dm(amplitudes)
        sleep(0.5)
        if prompt:
            input()
        
    amplitudes[z-1] = 0.0

def sweep_all_modes(**kwargs):
    for z in trange(1, Nmodes+1):
        sweep_mode(z, **kwargs)
    sleep(1)
    probe_signal(save=False)

def sweep_mode_combinations(**kwargs):
    patterns = 0.1 * np.array(
        reduce(lambda x, y: x + y, [
            [[float(i == j or i == j + k) for i in range(Nmodes)] for j in range(Nmodes - k)] for k in range(3)
        ])
    )
    for pattern in tqdm(patterns):
        command_to_dm(pattern)
        sleep(0.5)

    send_zeros()

def save_wait(save, wait):
    if save:
        save_telemetry(wait)
    else:
        sleep(wait)

def probe_signal(wait=0):
    send_zeros()
    # save_wait(save, wait)
    command_to_dm([0.9] + [0] * 11)
    sleep(0.5)
     #save_wait(save, wait)
    send_zeros()
    # save_wait(save, wait)

def random_combinations(Niters):
    inputzs = np.random.uniform(-1, 1, (Niters, Nmodes))
    np.save(f"../data/pl_230602/inputzs_{datetime.now().strftime('%H_%M_%S')}.npy", inputzs)
    for amp in tqdm(inputzs):
        command_to_dm(amp)

# %%
for i in trange(60):
    save_telemetry()
    sleep(5)
    # %%
