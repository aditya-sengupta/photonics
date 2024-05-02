using NPZ
using Plots
using PhotonicLantern

p_i_all = npzread("data/pl_231012/linearity_intensities_231012_1633.npy")
amps = npzread("data/pl_231012/linearity_amplitudes_231012_1633.npy")

for nmodes in [6, 12, 18]
    command_matrix = make_command_matrix(amps, p_i_all, nmodes=nmodes)
    plot_linearity(amps, p_i_all, command_matrix, nmodes=nmodes)
    Plots.savefig("figures/231012_linearity_$nmodes.png")
end
