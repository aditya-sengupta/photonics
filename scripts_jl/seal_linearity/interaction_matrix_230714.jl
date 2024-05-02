using NPZ
using Plots
using PhotonicLantern

amps = npzread("data/pl_230714/linearity_amplitudes_230714_1645.npy")
p_i_all = npzread("data/pl_230714/linearity_intensities_230714_1645.npy")

for nmodes in [6, 12, 18]
    command_matrix = make_command_matrix(amps, p_i_all, nmodes=nmodes)
    plot_linearity(amps, p_i_all, command_matrix, nmodes=nmodes)
    Plots.savefig("figures/230714_linearity_$nmodes.png")
end
