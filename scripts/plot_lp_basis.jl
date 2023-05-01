using Plots
using NPZ
using Measures

q = transpose(npzread("data/lp_basis.npy"))
begin
    p = plot(layout=grid(3,2),leg=false,framestyle=:none)
    for k = 1:6
        plot!(p[k], transpose(reshape(real.(q[:,k]), (160,160))), seriestype=:heatmap,ratio=:equal)
    end
    p
end

atm = npzread("data/input_fields_from_test_atm.npy")
k = 1
rk = vcat(transpose(atm[k,:,:])...)
q \ rk

