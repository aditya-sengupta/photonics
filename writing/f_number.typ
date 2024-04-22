#import "template.typ": *

#show: project.with(
  title: "f/# optimization for linear wavefront reconstruction with the photonic lantern",
  authors: (
    "Aditya Sengupta",
  ),
)

= Photonic lantern simulation overview

We simulate a 19-port photonic lantern using the _lightbeam_ package, with design parameters from private communication with Jon Lin. We set up a lantern "in reverse", where the input end is 8 times larger than the output. This enables us to propagate the fundamental mode from each SMF output and create a set of lantern modes. The performance of the PL as a WFS depends on how well the input PSF can be described as a linear combination of these modes.

#figure(
  image("../figures/lantern_modes_19.png", width: 75%),
  caption: [The modes created at the input of the photonic lantern by back-propagating from each single-mode fiber.],
) <lantern_modes_19>

To assess potential WFS performance as a function of the f-number at the input (which sets the PSF size), we can assess the coupling efficiency. We compute this following Lin 2021, by computing an overlap integral between an input PSF and its projection onto the lantern basis.

$ "coupling" = (integral "PSF" dot "projection" dif A) / sqrt(integral "PSF"^2 dif A integral "projection"^2 dif A) $

This is 1 if the PSF is fully described by the projection, and goes down if the component of the PSF not described by the lantern basis (light not coupled into the lantern) is larger. We compute this over a range from 0.1 to 20.

#figure(
  image("../figures/coupling_efficiency.png", width: 75%),
  caption: [The coupling efficiency as computed above, as a function of the f-number.],
) <coupling_efficiency>

This tells us that the best wavefront sensing performance can likely be achieved in the range f/4-8. Below this, we experience losses from drastically underfilling the input of the lantern, and above this, we overfill the input with the PSF core and are unlikely to be sensitive to significant aberrations, leading to large amounts of crosstalk and small linear ranges.

Zooming in on this range, we compute linearity curves and a loss value. The linearity curves are computed by creating an interaction matrix, inverting it to a command matrix, and recording the response induced by a range of amplitudes in each Zernike mode. The loss value is

$ "loss" =  sum_(i = 1)^(N_"Zernike") sum_(j = 1)^(N_"amplitude") abs("response"(i, j)_k - a(j) delta_(i k))^2 $

or, in words, the sum over squared deviations between the recorded response and the ideal linear response. This does not weight pokes close to 0 more strongly; the loss function could be adjusted to do this, but I think it's sufficient to just restrict the range of amplitudes we look at. 

Computing this over 9 Zernike modes, we find an optimum of $f slash 6.4$.

#figure(
  image("../figures/optimal_f.png", width: 100%),
  caption: [The linear reconstruction loss as a function of the f-number.],
) <optimal_f>

At $f slash 6.4$, we get a coupling efficiency of 99.7%, which is very good. Its low linearity loss is likely due to the PSF of that size being well represented by the lantern basis, while being sufficiently small that we don't overfill the lantern entrance and crosstalk starts to dominate.

We can observe how well the linear ranges behave as a function of the f-number:

#figure(
  image("../figures/linearity_fsweep_z9.png", width: 80%),
  caption: [The linear ranges as a function of the f-number.],
) <linearity_fsweep_z9>

The optimal f-number for more modes is likely to be smaller. We want the $(("cladding diameter") / ("spot size"))^2$ to be about the number of modes, which for a cladding diameter of $37.6 mu$m and a spot size of $f lambda slash D$, lets us go up to 

$ ((37.6) / (1.55f slash \#))^2 = N_"modes" arrow.r f slash \# = (37.6) / (1.55 sqrt(N_"modes")) $

For 9 modes this is $f slash 8$, which closely agrees with where the linearity begins to get erratic (there's likely some errors/inconsistencies from varying the f-number such that the spot size changes are smaller than the grid resolution). For 6 modes this is $f slash 9.9$ and for 18 modes this is $f slash 5.7$. I can rerun all of this for 6 and 18 Zernike modes but computing the linear ranges takes a while so I'm just reporting this for now.