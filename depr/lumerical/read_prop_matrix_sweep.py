# %%
import numpy as np
import photonics
from os import path

# %%
def port_to_index(port_num, mode_num):
    if port_num == 1:
        return 6 + mode_num # port 1 mode 1 is index 6, port 1 mode 2 is index 7, etc
    else:
        return port_num - 2 # port 2 is index 0, port 3 is index 1, etc
# %%
def frequency_to_matrix(fname, freq):
    
    port_indices = []
    matrix_elements = []
    with open(path.join(photonics.DATA_PATH, "lumerical", fname), 'r') as f:
        for line in f:
            if line.startswith('("port'):
                p = line.split(",")
                p1 = int(p[0][6:-1])
                m1 = int(p[2])
                p2 = int(p[3][6:-1])
                m2 = int(p[4])
                port_indices.append((port_to_index(p1, m1), port_to_index(p2, m2)))
            elif line.startswith(freq):
                s = line.split(" ")
                matrix_elements.append(float(s[1]) * np.exp(1j * float(s[2])))

    n = np.max([x[1] for x in port_indices]) + 1
    S = np.zeros((n, n), dtype=np.complex128)
    for (inds, el) in zip(port_indices, matrix_elements):
        S[inds[0], inds[1]] = el

    return S
# %%
S_short = frequency_to_matrix("melting_short_sweep.dat", "4.7211410708661419e+14")
# %%
S_long = frequency_to_matrix("melting_long_sweep.dat", "1.9341448903225806e+14")
# %%
plt.imshow(np.abs(S_short[:7, 7:]))
plt.title("Abs(S) for 635nm melting, starting at 635nm")
plt.xlabel("MMF mode number")
plt.ylabel("SMF port index")
plt.colorbar()
# %%
plt.imshow(np.abs(S_long[:7, 7:]))
plt.title("Abs(S) for 1.55um melting, starting at 1.55um")
plt.xlabel("MMF mode number")
plt.ylabel("SMF port index")
plt.colorbar()
# %%
plt.imshow(100 * np.abs((S_short[:7, 7:] - S_long[:7, 7:]) / (S_long[:7, 7:])))
plt.title("Abs(% change) long to short, starting at 1.55um")
plt.xlabel("MMF mode number")
plt.ylabel("SMF port index")
plt.colorbar()
# %%
