using NPZ
using LinearAlgebra
using Chain
ENV["JULIA_PYTHONCALL_EXE"] = "./.venv/bin/python"
using PythonCall
using Plots
lightbeam = pyimport("lightbeam")

function make_unitary_matrix(N)
    H = rand(ComplexF64, N, N)
    exp(1im * (H + H'))
end

function is_unitary(M)
    M * M' ≈ I && M' * M ≈ I
end

atm_screens = npzread("data/input_fields_from_test_atm.npy");
imshow