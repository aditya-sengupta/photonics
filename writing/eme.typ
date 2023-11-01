#import "template.typ": *

#show: project.with(
  title: "Eigenmode expansion for photonic lanterns",
  authors: (
    "Aditya Sengupta",
  ),
)

= Motivation

The finite-difference beam propagation method (FD-BPM) of simulating a photonic lantern is a lot less efficient than what we need for photonic lantern simulations, difficult to connect to physical interpretations, and hard to optimize over due to the discretization in each direction. We can qualitatively understand its ineffiency by noting that

- FD-BPM doesn't have a preferred direction, as it's meant primarily for free-space cavities rather than waveguides, so it isn't able to optimize as well along the direction of propagation;
- the PL spreads out with increasing $z$, leading to more regions of empty space/where we don't care about the fields but where we're forced to compute them anyway;
- we end up only needing as many output intensities as there are ports, so we compute a lot of high-resolution information just to average it out at the end

I found that the preferred method for waveguides such as optical fibers and MMIs, where the size along the direction of propagation is much larger than that perpendicular to it, is _eigenmode expansion_ (EME). This involves expressing the electric field as a linear combination of eigenmodes, and propagating the coefficients of those eigenmodes through the waveguide. This has the advantage of being an exact solution -- the number of components is just the number of eigenmodes, which is limited by the size of the input port relative to the wavelength of light. Further, the method's integration error is limited by the cell resolution along the direction of propagation, and after the point where mode profiles don't overlap substantially, we can treat each port the same as it has the same index of refraction profile and is unaffected by other ports. 

Unfortunately, the only implementations or descriptions of EME I can find have been worked out by private CAD companies and so the details are unavailable. I'm going to fill in some of those details here so we can do this!

= Cylindrical waveguides

In order to establish the theory we're working with, let's derive the eigenmode expansion for a cylindrical waveguide first. We'll start with the time-independent wave equation in cylindrical coordinates:

$ (diff^2 E) / (diff r^2) + 1 / r (diff E) / (diff r) + 1 / r^2 (diff^2 E) / (diff phi^2) + (diff^2 E) / (diff z^2) + k^2 E = 0. $

Here, $k = (2 pi n) / lambda$ is the wavenumber. $n$ can vary with $r$, and we might eventually also have to make it vary with $z$, but we'll have radial symmetry throughout. Separating variables lets us pull out the $phi$ component as $exp(plus.minus i l phi)$ and the $z$ component as $exp(i beta_(l m) z)$, for an eigenvalue $beta_(l m)$.d

/*
We're trying to solve Laplace's equation, $nabla^2 phi = 0$, which we can expand in cylindrical coordinates:

$ 1 / rho diff / (diff rho) (rho (diff phi) / (diff rho)) + 1 / rho^2 (diff^2 phi) / (diff phi.alt^2) + (diff^2 phi) / (diff z^2) = 0 $

and we can separate this by variable, $phi(rho, phi.alt, z) = R(rho) G(phi.alt) Z(z)$, to get

$ rho dif / (dif rho) (rho (dif R) / (dif rho)) + (k^2 rho^2 - alpha^2) R = 0 $
$ (dif^2 G) / (dif phi.alt^2) + alpha^2 G = 0 $
$ (dif^2 Z) / (dif z^2) - k^2 Z = 0 $

Further, continuity on $phi.alt$, i.e. requiring that $G(0) = G(2pi)$ and that $G'(0) = G'(2pi)$, gives us $alpha = n$ where $n >= 0$ is an integer. */
