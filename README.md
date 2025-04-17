# GRaTeR-JAX

**GRaTeR-JAX** is a machine learning JAX-based implementation of the **Generalized Radial Transporter (GRaTeR)** framework, designed for modeling scattered light disks in protoplanetary systems. This repository provides tools for forward modeling, optimization, and parameter estimation of scattered light disk images using JAX's accelerated computations.

## Features

- **JAX-Based Optimization**: Leverages JAX for fast, GPU/TPU-accelerated disk modeling.
- **Scattered Light Disk Modeling**: Implements physical models of exoplanetary debris disks.
- **Differentiable Framework**: Enables gradient-based optimization and probabilistic inference.
- **Integration with Webbpsf**: Supports PSF convolution for telescope observations.

## Installation

To install GRaTeR-JAX and its dependencies, run:

```sh
git clone https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX.git
cd GRaTeR-JAX
pip install -r requirements.txt
```

Make sure you have JAX installed with the correct backend for your hardware:

```sh
pip install --upgrade "jax[cpu]"  # or "jax[cuda]" for GPU
```

## Usage

### Refer to ObjectiveFunctionTest.ipynb for building basic disk models and fitting them to images.

Information about the disk and misc parameters can be found in objective_functions.py. Information about the
scattering phase function and point spread function parameters can be found in SLD_utils.py.

## Repository Structure

```
GRaTeR-JAX/
│── utils/                 # Utility functions for modeling and optimization
│── statistical-analysis/  # Example Jupyter notebooks
│── webbpsf-data           # PSF data for various instruments
│── cds/                   # Utility files for WebbPSF convolution
│── PSFs/                   # PSF data
│── requirements.txt       # Dependencies
│── README.md              # This document
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Commit your changes and push to your fork.
4. Open a pull request.

## Acknowledgments

Developed by the **UCSB Exoplanet Polarimetry Lab**. This work is inspired by previous implementations of GRaTeR and advances in JAX-based differentiable modeling.

---
