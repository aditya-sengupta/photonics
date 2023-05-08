# %%
import lightbeam
# %%
taper_factor = 4
rcore, rclad, rjack = 2.2 / taper_factor, 4.0, 0.0
zw = 10000
nclad = 1.4504
ncore = nclad + 0.0088
njack = nclad - 5.5e-3
nvals = (ncore, nclad, njack)

lantern = lightbeam.make_lant3big(rclad / 2, rcore, rclad, rjack, zw, nvals, final_scale=taper_factor)
# %%
lantern.check_smfs(2*np.pi/5)
# %%
