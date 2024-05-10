using FFTW
using Plots

f_loop = 100.0
f_cutoff = 10.0
f_fast = 40.0
f_slow = 5.0
α = exp(-f_cutoff / f_loop)
t = 0:(1/f_loop):5
N = length(t)
a_fast, a_slow = 0.5, 3.0

cn = rand(N)
# cn = @. a_fast * sin(2π * f_fast * t) + a_slow * sin(2π * f_slow * t) 
cHn = zeros(N)
cLn = zeros(N)

for i in 2:N
    cHn[i] = α * cHn[i - 1] + α * (cn[i] - cn[i - 1])
    cLn[i] = α * cLn[i - 1] + (1 - α) * (cn[i - 1])
end

F = fftshift(fft(cn))[N÷2+1:end]
FH = fftshift(fft(cHn))[N÷2+1:end]
FL = fftshift(fft(cLn))[N÷2+1:end]
freqs = fftshift(fftfreq(length(t), f_loop))[N÷2+1:end]

begin
    plot(freqs, (@. abs2(FH)/abs2(F)), label="High-frequency rejection TF")
    plot!(freqs, (@. abs2(FL)/abs2(F)), label="Low-frequency rejection TF")
end