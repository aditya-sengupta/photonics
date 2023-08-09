import subprocess
from astropy.io import fits
from datetime import datetime
from functools import reduce
from time import sleep
from tqdm import tqdm, trange

if False: # post-trip don't accidentally run this!
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
            if prompt:
                sleep(0.5)
                input()
            
        amplitudes[z-1] = 0.0

    def sweep_all_modes(**kwargs):
        for z in trange(1, Nmodes+1):
            sweep_mode(z, **kwargs)
        sleep(1)
        probe_signal()

    def sweep_mode_combinations(amp=0.1, kmax=11, **kwargs):
        patterns = amp * np.array(
            reduce(lambda x, y: x + y, [
                [[float(i == j or i == j + k) for i in range(Nmodes)] for j in range(Nmodes - k)] for k in range(kmax)
            ])
        )
        for pattern in tqdm(patterns):
            command_to_dm(pattern)

        send_zeros()

    def probe_signal(wait=0):
        send_zeros()
        command_to_dm([0.9] + [0] * 11)
        sleep(0.5)
        send_zeros()

    def random_combinations(Niters):
        time_stamps = []
        inputzs = np.random.uniform(-1, 1, (Niters, Nmodes))
        start_stamp = datetime.now().strftime('%H_%M_%S')
        np.save(f"../data/pl_230603/inputzs_{start_stamp}.npy", inputzs)
        for (i, amp) in enumerate(tqdm(inputzs)):
            t = datetime.now()
            time_stamps.append([i, t.hour, t.minute, t.second, t.microsecond])
            command_to_dm(amp)

        np.save(f"../data/pl_230603/timestamps_{start_stamp}.npy", np.array(time_stamps))

    probe_signal()
    sweep_mode_combinations(amp=1.0)
    probe_signal()
        # %%
    probe_signal()
    sweep_all_modes()
    probe_signal()
    sweep_all_modes()
    probe_signal()
    # %%
