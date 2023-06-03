using Flux
using Flux: train!
using PhotonicLantern
using ProgressMeter

Nports = 18
Nouts = 18
Nex = 1000
A = rand(Nouts, Nports)

actual(x) = abs2.(A * cos.(x))
x_train, x_test = rand(Nports, Nex), rand(Nports, Nex)
cosx_train, cosx_test = cos.(x_train), cos.(x_test)
y_train, y_test = hcat(actual.(eachcol(x_train))...), hcat(actual.(eachcol(x_test))...)
model = Chain(
    Dense(Nports => Nouts, abs2),
)
opt = Flux.setup(Adam(0.1), model)
data = [(cosx_train, y_train)]

loss(model, x, y) = Flux.mse(model(x), y)
loss(model, cosx_train, y_train)

@showprogress for _ in 1:2000
    train!(loss, model, data, opt)
end

loss(model, cosx_train, y_train)
loss(model, cosx_test, y_test)