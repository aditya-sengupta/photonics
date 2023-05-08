using Flux, ProgressMeter
using PhotonicLantern
using Plots
using Distributed

nports = 6
ntest = 1000
U = make_unitary_matrix(nports)
input_phase = 2 * rand(Float64, nports, ntest) .- 1
input_field = exp.(1im .* input_phase)
output_intensity = hcat([abs2.(U * c) for c in eachcol(input_field)]...)

model = Chain(
    Dense(nports => 2*nports, bias=false)
)

optim = Flux.setup(Flux.Adam(0.1), model)

function get_complex_loss(u_out, u_pre)
    l = 0.0
    for (o, p) in zip(eachcol(u_out), eachcol(u_pre))
        for i in 1:nports
            l += abs2(o[i] - abs2(p[i] + 1im * p[i + nports]))
        end
    end
    l
end

get_complex_loss(output_intensity, model(input_field))

losses = []
begin
    @showprogress for epoch in 1:100
        loss, grads = Flux.withgradient(model) do m 
            get_complex_loss(output_intensity, m(input_field))
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end
plot(losses)

A = optim.layers[1].weight.state[1]
A = A[1:nports,:] + 1im * A[(nports+1):2*nports,:]
U