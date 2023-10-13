#import "template.typ": *

#show: project.with(
  title: "Converting focal-Zernike modes to LP modes",
  authors: (
    "Aditya Sengupta",
  ),
)

We're interested in taking the Zernike modes, converting them into aberrated PSFs at the focal plane, and seeing how those couple into the LP modes of a photonic lantern. Both the Zernikes and the step-index LPs have sufficiently "nice" representations that this should be possible semi-analytically, but we'll need to resolve the lantern LP modes numerically, since it's not radially symmetric. So we'll just do this numerically for now, and think about other stuff later.