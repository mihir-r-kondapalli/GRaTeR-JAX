# GRaTeR-JAX

**GRaTeR-JAX** is a JAX-based implementation of the **Generalized Radial Transporter (GRaTeR)** framework, designed for modeling scattered light disks in protoplanetary systems. This repository provides tools for forward modeling, optimization, and parameter estimation of scattered light disk images using JAX's accelerated computations.

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

### Example: Running a Basic Scattered Light Disk Model

```python
import jax.numpy as jnp
from utils.SLD_ojax import compute_disk_model

params = {"inclination": 75, "scale_height": 0.1, "albedo": 0.5}
disk_image = compute_disk_model(params)
```

### PSF Convolution Example

```python
from utils.PSFConv import convolve_with_psf
psf_convolved_disk = convolve_with_psf(disk_image, psf_kernel)
```

### Example: Parameter Estimation with JAX Optimization

```python
from jax import grad, jit
from utils.objective_functions import loss_function

@jit
def optimize_params(params):
    grad_loss = grad(loss_function)
    return params - 0.01 * grad_loss(params)  # Simple gradient descent step
```

### Example: Using WebbPSF Data

```python
from webbpsf import instrument

jwst = instrument.JWST()
jwst.load_wavelength_dependent_psf()
```

## Repository Structure

```
GRaTeR-JAX/
│── utils/                 # Utility functions for modeling and optimization
│── statistical-analysis/  # Example Jupyter notebooks
│── webbpsf-data           # PSF data for various instruments
│── cds/                   # Utility files for WebbPSF convolution
│── PSF/                   # Empirical PSFs
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
