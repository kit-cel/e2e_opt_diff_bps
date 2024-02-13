# End-to-end optimization of Constellation Shaping for Wiener Phase Noise Channels with a Differentiable Blind Phase Search

This repository accompanies methods and systems presented in [\[1\]](https://ieeexplore.ieee.org/document/10093964/) on the topic of end-to-end optimization of geometric and probabilistic constellation shaping for the Wiener phase noise channel. In this repository we publish application scripts and notebooks which perform the optimization. In order to successfully execute the applications we rely on functions and modules implemented in our own machine learning and optimization library [mokka](https://github.com/kit-cel/mokka).

## Usage

First install the [mokka](https://github.com/kit-cel/mokka) library to a Python virtualenvironment.
To train a geometrically shaped constellation execute the application `./apps/BPS_autoencoder.py`. Use the `-h` flag to receive a list of available commandline options.

For performance evaluation execute the application `./apps/BPS_performance.py`. With the `-h` flag a list of available commandline options is shown.


## References
[\[1\] A. Rode, B. Geiger, S. Chimmalgi, and L. Schmalen, ‘End-to-end optimization of constellation shaping for Wiener phase noise channels with a differentiable blind phase search’, Journal of Lightwave Technology, pp. 1–11, 2023, doi: 10.1109/JLT.2023.3265308.](https://ieeexplore.ieee.org/document/10093964/)

## Acknowledgment
This  work  has  received  funding  from  the  European  Re-search Council (ERC) under the European Union's Horizon2020 research and innovation programme (grant agreement No. 101001899).
