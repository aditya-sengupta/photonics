using NPZ
using Plots
using PhotonicLantern

p_i_all = npzread("data/pl_231020/linearity_responses.npy")
amps = npzread("data/pl_231020/linearity_amps.npy")

command_matrix = make_command_matrix(amps, p_i_all, nmodes=10, push_ref=0.05)
plot_linearity(amps, p_i_all, command_matrix, nmodes=10)
Plots.savefig("figures/seal_linearity.png")
plot_linearity(amps, p_i_all, command_matrix, nmodes=6)
Plots.savefig("figures/linear_with_axes_231023.png")
