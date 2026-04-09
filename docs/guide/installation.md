```md
## Installation

We recommend using Conda to create a fresh environment in which to run GRaTer-JAX, to ensure all dependencies are met:

```bash
conda env create -f environment.yml
conda activate grater-jax-env   # or your env name
pip install -e .
```

To install GRaTeR-JAX and its dependencies, create a new Conda environment with Python and run:

```pip install grater-jax```

Make sure you have JAX installed with the correct backend for your hardware:

```pip install --upgrade "jax[cpu]"  # or "jax[cuda]" for GPU```

NOTE: For Millar-Blanchaer Lab developers -- you need to install with Jax 0.4.34 and Cuda 12.2 if using Mueller.

## Compatibility

GRaTer-JAX is compatible with Python >=3.9 and <3.13, and any GPU processor supported by JAX (see [the JAX install docs](https://docs.jax.dev/en/latest/installation.html) for a list). A full list of dependencies is available in `pyproject.toml`

