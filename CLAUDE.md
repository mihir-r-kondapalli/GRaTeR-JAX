# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRaTeR-JAX is a JAX-accelerated Python package for modeling scattered-light observations of circumstellar debris disks. It implements the GRaTeR (Generalized Radial Transporter) framework using automatic differentiation (via JAX) to enable gradient-based optimization and Bayesian inference.

## Development Setup

```bash
# Install with development dependencies (includes pytest and vip_hci for testing)
pip install --upgrade jax[cpu]
pip install -e .[dev]

# For JWST PSF support (optional, experimental)
pip install -e .[webbpsf]

# For building documentation
pip install -e .[docs]
```

## Commands

```bash
# Run all tests
cd tests && pytest

# Run a single test file
cd tests && pytest test_model.py

# Run a single test function
cd tests && pytest test_model.py::test_function_name

# Build documentation (from docs/)
cd docs && make html

# Build package for release
python -m build
twine check dist/*
```

Tests use Python 3.11 in CI (`.github/workflows/tests.yml`). No linting is configured.

## Architecture

The package has two main subsystems:

### 1. Disk Modeling (`grater_jax/disk_model/`)

Components are designed to be modular and swappable. The core flow is:

```
Parameters → ScatteredLightDisk → forward model → convolved image
```

**Key classes:**
- **`SLD_ojax.py` — `ScatteredLightDisk`**: Main disk model. Generates synthetic scattered-light images given geometry (inclination, PA, eccentricity), dust density distribution, and scattering phase function.
- **`SLD_utils.py`**: Modular components with a common `Jax_class` base:
  - Dust distributions: `DustEllipticalDistribution2PowerLaws`
  - Scattering phase functions (SPFs): `HenyeyGreenstein_SPF`, `DoubleHenyeyGreenstein_SPF`, `InterpolatedUnivariateSpline_SPF`
  - PSF models: `GAUSSIAN_PSF`, `EMP_PSF`, `Winnie_PSF`
  - Stellar PSF models: `LinearStellarPSF`, `PositionalStellarPSF`
- **`winnie_class.py` — `WinniePSF`**: JAX-compatible wrapper for spatially-varying PSF convolution with roll-angle support. Implements the Winnie package as a JAX PyTree.
- **`jax_model_wrappers.py`**: `@jax.jit`-compiled forward model functions. Each specialization handles a different combination of PSF/SPF types.
- **`objective_functions.py`**: Bridges models to optimization — packs/unpacks parameter dicts to JAX arrays, computes log-likelihoods and gradients. Contains `Parameter_Index` with default parameter templates.

### 2. Optimization (`grater_jax/optimization/`)

- **`optimize_framework.py` — `Optimizer`**: High-level API integrating all components. Entry point for users. Provides:
  - `get_model()`: Generate model images
  - `get_objective_likelihood()`: Log-likelihood evaluation
  - `get_objective_gradient()`: Gradient computation via JAX autodiff
  - `scipy_optimize()` / `scipy_bounded_optimize()`: Deterministic optimization
  - `mcmc()`: Bayesian inference with emcee
- **`mcmc_model.py` — `MCMC_model`**: Wraps `emcee` ensemble sampler with helpers for extracting posterior statistics and producing corner/chain plots.

### JAX Design Patterns

- All JAX-traceable classes register as PyTrees to support `jit`, `vmap`, and `grad`
- Parameter dictionaries are packed into flat JAX arrays for optimization routines
- The `Jax_class` base in `SLD_utils.py` provides the pack/unpack interface used throughout

### Tests

Tests in `tests/` validate:
- `test_model.py`: Numerical agreement between JAX implementation and original VIP (`vip_hci`) reference
- `test_model_gradient.py`: Gradient correctness
- `test_optimizer.py`: End-to-end optimizer functionality
- `test_sld_utils.py`: Individual component utilities

## Known Issues

- `docs/index.md` contains unresolved Git merge conflict markers (lines 8–19)
- JWST PSF support (`winnie_jwst_fm.py`) is marked experimental
