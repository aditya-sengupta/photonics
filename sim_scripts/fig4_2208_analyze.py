# %%
from fig4_2208_config import *
from lightbeam.misc import norm_nonu, overlap_nonu
from tqdm import tqdm
from os.path import join
import re
# %%
if __name__ == "__main__":
    t = 2 * np.pi / 5
    core_locs = [[0.0,0.0]] + [[2 * np.cos(i*t), 2 * np.sin(i*t)] for i in range(5)]
    w = mesh.xy.get_weights()
    # %%
    u = np.load(join(ROOT_DIR, "data/zerns/2208_4_2_-0.6.npy"))
    output_powers = []
    for pos in core_locs:
        _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
        output_powers.append(np.power(overlap_nonu(_m,u,w),2))
    # %%
    powers = np.empty((5,11,6)) # input zernikes x amplitudes x ports
    ampls = np.empty((5,11)) # input zernikes x amplitudes
    input_zerns = list(range(1,6))
    files = os.listdir(join(ROOT_DIR, "data/zerns"))
    for z in input_zerns:
        zfiles = list(filter(lambda x: x.startswith(f"2208_4_{z}"), files))

    for z in input_zerns:
        zfiles = list(filter(lambda x: x.startswith(f"2208_4_{z}"), files))
        for (j, f) in enumerate(tqdm(zfiles)):
            ampl = float(re.match(fr"2208_4_{z}_(\S+).npy", f).group(1))
            ampls[z-1][j] = ampl
            u = np.load(join(ROOT_DIR, f"data/zerns/{f}"))
            for (i, pos) in enumerate(core_locs):
                _m = norm_nonu(lpfield(mesh.xg-pos[0],mesh.yg-pos[1],0,1,rcore*scale,prop.wl0,ncore,nclad),w)
                powers[z-1][j][i] = np.power(overlap_nonu(_m,u,w),2)
    # %%
    np.save(join(ROOT_DIR, "data/zerns/2208_4_ampls.npy"), ampls)
    np.save(join(ROOT_DIR, "data/zerns/2208_4_powers.npy"), powers)
    # %%
    ampls = np.load(join(ROOT_DIR, "data/zerns/2208_4_ampls.npy"))
    powers = np.load(join(ROOT_DIR, "data/zerns/2208_4_powers.npy"))
    powers = np.nan_to_num(powers, 0)
    colors = ['r', 'g', 'b', 'k', 'c', 'm']
    fig, axs = plt.subplots(5,1, figsize=(2,10), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    for j in range(5):
        perm = np.argsort(ampls[j])
        print(powers[j,:,:][perm])
        for i in range(6):
            axs[j].plot(ampls[j][perm], powers[j,:,i][perm], colors[i], label=f"Port {i}")
        axs[j].set_title(f"Zernike mode {j+1}")
            
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel("Input amplitude")
    plt.ylabel("Output response")
    plt.savefig(join(ROOT_DIR, "figures/2208_4_repr.pdf"), bbox_inches='tight')
    # %%