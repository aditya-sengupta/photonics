using NPZ
using Plots
using PhotonicLantern

p_i_all = npzread("data/pl_231025/linearity_responses.npy")# * rad2opd
amps = npzread("data/pl_231025/linearity_amps.npy")# * rad2opd

command_matrix = make_command_matrix(amps, p_i_all, nmodes=10, push_ref=0.05)
plot_linearity(amps, p_i_all, command_matrix, nmodes=10)
plot_linearity(amps, p_i_all, command_matrix, nmodes=6)
