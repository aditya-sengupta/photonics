using Plots

f_numbers = [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 6, 7, 8, 9, 10]

loss_values = [0.010555671, 0.010714758, 0.010817282, 0.012359229, 0.011512046, 0.0106320605, 0.01034948, 0.0103400685, 0.010478149, 0.010357336, 0.010503123, 0.010452948, 0.010778161, 0.0109749865, 0.011001634, 0.0106940875, 0.010911635, 0.010830192, 0.010914187, 0.010927338, 0.011694369, 0.009712769, 0.009601091, 0.009832502, 0.009708853, 0.009806862]

plot(f_numbers, loss_values)