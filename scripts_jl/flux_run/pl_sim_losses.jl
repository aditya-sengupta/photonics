using NPZ
using Plots

hidden_layer_size = 20:-1:0

begin
    p = plot(xlabel="Hidden layer size", ylabel="Final loss (rad RMSE)", yscale=:log10)
    colors = palette([:red, :blue], 9)
    for f in 3:11
        d = npzread("data/pl_nn/test_losses_f_$f.npy")
        for i in eachindex(d)
            d[i] = maximum(d[1:i])
        end
        plot!(hidden_layer_size, d, label="f/$f", c=colors[f-2])
    end
    p
end