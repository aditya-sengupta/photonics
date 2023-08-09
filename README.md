# photonics

This repository contains resources for working with photonic lanterns, and experimental results. It is (should be) installable from GitHub both as a Python package and as a Julia package.

Source code is in:
- `photonics`: Python source files describing the LanternReader class and basic operations with SEAL/Shane.
- `src`: Julia source (mostly utility functions); if you're not Aditya you can probably ignore this.
- `scripts`: .py and .jl files that can be run from the top level.
    - `analysis`: scripts to analyze data, fit reconstruction algorithms/neural networks, etc.
    - `onsky`: scripts to run at Shane.
    - `seal`: scripts to run on the SEAL testbed.
    - `sim`: scripts to run simulations.

Data is saved to `data` and figures are saved to `figures`.

The simulation parts of this package depends on Aditya's version of `lightbeam`, at https://github.com/aditya-sengupta/lightbeam. For the purpose of running Shane experiments, this should be safe to ignore.