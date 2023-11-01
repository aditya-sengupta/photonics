#import "template.typ": *

#show: project.with(
  title: "Lantern signal-to-noise on SEAL",
  authors: (
    "Aditya Sengupta",
  ),
)

I'm trying to measure the signal-to-noise ratio on each of the lantern ports on SEAL. We have

- read noise: proportional to the number of pixels in each port
- dark noise: proportional to the exposure time, can be measured

and we're able to measure a number of counts per pixel within each lantern port, which represents the sum of all these things. 