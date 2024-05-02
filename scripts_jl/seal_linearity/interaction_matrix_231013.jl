using NPZ
using Plots
using PhotonicLantern

p_i_all = npzread("data/pl_231013/linearity_intensities_231013_1516.npy")
amps = npzread("data/pl_231013/linearity_amplitudes_231013_1516.npy")

command_matrix = make_command_matrix(amps, p_i_all, nmodes=nmodes, push_ref=0.1)
plot_linearity(amps, p_i_all, command_matrix, nmodes=nmodes)
Plots.savefig("figures/231013_linearity_$nmodes.png")