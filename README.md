GRaTeR-JAX

GRaTeR-JAX is a JAX-based implementation of the Generalized Radial Transporter (GRaTeR) framework, designed for modeling scattered light disks around exoplanets. This repository provides tools for forward modeling, optimization, and parameter estimation of scattered light disk images using JAX's accelerated computations.

Features

JAX-Based Optimization: Leverages JAX for fast, GPU/TPU-accelerated disk modeling.

Scattered Light Disk Modeling: Implements physical models of exoplanetary debris disks.

Differentiable Framework: Enables gradient-based optimization and probabilistic inference.

Integration with Webbpsf: Supports PSF convolutions for telescope observations.

Installation

To install GRaTeR-JAX and its dependencies, run:

git clone https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX.git
cd GRaTeR-JAX
pip install -r requirements.txt

Make sure you have JAX installed with the correct backend for your hardware:

pip install --upgrade "jax[cpu]"  # or "jax[cuda]" for GPU

Usage

Example: Running a Basic Scattered Light Disk Model

import jax.numpy as jnp
from utils.SLD_ojax import compute_disk_model

params = {"inclination": 75, "scale_height": 0.1, "albedo": 0.5}
disk_image = compute_disk_model(params)

PSF Convolution Example

from utils.PSFConv import convolve_with_psf
psf_convolved_disk = convolve_with_psf(disk_image, psf_kernel)

Repository Structure

GRaTeR-JAX/
│── utils/                 # Utility functions for modeling and optimization
│── statistical-analysis/  # Example Jupyter notebooks
│── webbpsf-data           # psf data for various instruments
│── cds/                   # Utility files for webb psf convolutioin
│── PSF/                   # Empirical psfs
│── requirements.txt       # Dependencies
│── README.md              # This document

Contributing

We welcome contributions! To contribute:

Fork the repository.

Create a feature branch: git checkout -b feature-branch.

Commit your changes and push to your fork.

Open a pull request.

Acknowledgments

Developed by the UCSB Exoplanet Polarimetry Lab. This work is inspired by previous implementations of GRaTeR and advances in JAX-based differentiable modeling.