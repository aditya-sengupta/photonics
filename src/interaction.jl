using Plots
using LinearAlgebra: svd

mode_names = ["x-tilt", "y-tilt", "astig", "focus", "astig45", "tricoma", "tricoma60", "coma", "coma90", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19"]

function make_command_matrix(amps, p_i_all; push_ref=0.1, nmodes=Inf)
    l, r = argmin(abs.(amps .+ push_ref)), argmin(abs.(amps .- push_ref))
    p_i = p_i_all[:,[l,r],:]
    s1, s3 = size(p_i_all, 1), size(p_i, 3)
    s1 = Int(min(s1, nmodes))
    p_i = p_i[1:s1,:,:]
    z_i = zeros(s1, 2, s1)
    for k in 1:s1
        z_i[k,:,k] = [-push_ref, push_ref]
    end

    p = reshape(p_i, ((s1 * 2, s3)))
    z = reshape(z_i, ((s1 * 2, s1)))

    # A = z \ p
    command_matrix = p \ z
    return command_matrix
end

function plot_linearity(amps, p_i_all, command_matrix; nmodes=18)
    pl = []
    lim = maximum(amps)
    for k in 2:(nmodes+1)
        lin_k = ((p_i_all[k-1,:,:]) * command_matrix)[:,1:nmodes]
        a = [(k == i ? 1 : 0.2) for i in 2:(nmodes+1)]'
        pk = plot(xlabel=mode_names[k-1], label=nothing, legend=:outertopright, xticks=nothing, yticks=nothing, xlim=(-lim, lim), ylim=(-lim, lim))
        plot!(amps, lin_k, alpha=a)
        plot!(amps, amps, ls=:dash, color=:black)
        push!(pl, pk)
    end
    nrows = 3 # nmodes ÷ 6
    plot(pl..., legend=nothing, dpi=200, layout=(nrows, 3), size=(600,600))
    #  size=(200 * 6, 200 * nrows), 
end

export make_command_matrix, plot_linearity