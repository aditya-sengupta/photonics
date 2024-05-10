#import "template.typ": *

#show: project.with(
  title: "Understanding the autoregressive filter used for second-stage control",
  authors: (
    "Aditya Sengupta",
  ),
)

#set math.mat(delim: "[")

In his 2023 SPIE paper, Ben uses this filter to separate out the signals going into the SHWFS and FAST:

$ 
  c_(H{n}) &= alpha c_(H{n-1}) + alpha (c_n - c_(n-1)) \
  c_(L{n}) &= alpha c_(L{n-1}) + (1 - alpha) c_(n-1)
$

where $alpha = e^(-f_"cutoff" slash f_"loop")$. I'd like to convince myself that these do actually define high- and low-pass filters respectively, so I'll calculate their frequency responses. We'll treat $c_n$ as the filter input and $c_(H{n}), c_(L{n})$ as the outputs. Let $c_n = e^(i omega n)$ where $n = t f_"loop"$, let $c_(H{n}) = F_H (omega) e^(i omega n)$, and let $c_(L{n}) = F_L (omega) e^(i omega n)$.

We get

$ 
  F_H (omega) &= (alpha (1 - e^(-i omega))) / (1 - alpha e^(-i omega)) ; space F_L (omega) &= ((1 - alpha) e^(-i omega)) / (1 - alpha e^(-i omega))
$

and we're interested in the magnitude of this, so we multiply the numerators and denominators by $(1 - alpha e^(-i omega))^* = 1 - alpha e^(i omega)$, and take a magnitude of what's left over.

$ 
  |F_H (omega)| &= alpha / (1 + alpha^2) sqrt(1 + alpha^2 + 2alpha + (1 - alpha)^2 sin^2 omega)  \
  |F_L (omega)| &= (1 - alpha) / (1 + alpha^2) sqrt(1 + alpha^2 - 2 alpha cos omega)
$

If we remember to interpret $omega$ as $f slash f_"loop"$, because this is a discrete-time signal, we get curves that look correct!
#figure(
  image("../figures/writing/hpf_lpf.png", width: 90%),
  caption: [The theoretical frequency response of both filters.],
) <hpf_lpf>

If we simulate the filters in time and feed in a random array (representing all frequency components), we get similar curves:

#figure(
  image("../figures/writing/hpf_lpf_empirical.png", width: 90%),
  caption: [The empirical frequency response of both filters.],
) <hpf_lpf_empirical>

And looking at it in time, we see that the two signals have been separated out in frequency:

#figure(
  image("../figures/writing/filter_timeres.png", width: 90%),
  caption: [The time response of both filters.],
) <filter_timeres>

The high-pass signal is left with a significant contribution from the low frequencies, so I'm not convinced there won't be any crosstalk. Maybe this gets removed by closing the loop on the whole signal first, because integrators get rid of high-frequency noise, I think. 

We can take a $z$ transform by taking $z = e^(i omega)$:

$
  Z_H (z) = alpha (z - 1) / (z - alpha) \
  Z_L (z) = (1 - alpha) / (z - alpha)
$

If I'm putting this in the SPIE paper, I should make sure this is all done consistently in $s$ or $z$.

To test this, let's multiply these frequency responses by the frequency representation of the closed-loop signal on the pyramid. The initial pyramid closed loop looks at all modes, so there's no issues with overlapping the correct parts of parameter space with one another yet.

The transfer function for a closed loop on a plant $G(s)$ is given by $G(s) / (1 + G(s))$. If the plant is an integrator $G(s) = g slash s$, we have $(s) / (s + g)$. This doesn't change things too much.

