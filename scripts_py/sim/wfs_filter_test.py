# %%
import numpy as np
from photonics.wfs_filter import WFSFilter

f = WFSFilter(2, 0.5)
signal = np.random.randn(10, 2)

# %%
for s in signal:
    h, l = f(s)
    print(np.allclose(h + l, s))
# %%
