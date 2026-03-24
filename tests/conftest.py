"""
conftest.py
===========

Top-level pytest configuration for the GRaTeR-JAX test suite.

Prevents a segmentation fault caused by JAX's CUDA plugin
(xla_cuda_plugin.so) when CUDA runtime libraries (cuDNN, cuBLAS, etc.)
are not installed in the conda environment.  The fix runs *before* any
test file is imported, which is also before JAX is first imported.

Strategy
--------
Inject a no-op replacement for ``jax_plugins.xla_cuda12`` into
``sys.modules``.  When JAX's plugin-discovery code calls
``importlib.import_module('jax_plugins.xla_cuda12')``, Python finds our
fake module immediately (without touching the real .so file) and calls
our no-op ``initialize()``.  Setting ``JAX_PLATFORMS=cpu`` then directs
JAX to the CPU XLA backend.

Permanent fix
-------------
To restore GPU support, install the missing CUDA runtime libraries in
the conda environment::

    conda install -n grater-jax -c nvidia cudnn cublas

or reinstall JAX with CUDA extras::

    pip install --upgrade "jax[cuda12]"
"""
import os
import sys
import types

# Tell JAX to use the CPU backend (read at JAX import time).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Tests save plots to "test_results/" relative to CWD.  Create it so that
# plt.savefig() works regardless of which directory pytest is invoked from.
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "test_results"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "test_results"), exist_ok=True)

# Replace the CUDA plugin with a no-op *before* JAX is imported so that
# JAX's plugin-discovery loop never tries to dlopen the crashing .so.
_fake_cuda_plugin = types.ModuleType("jax_plugins.xla_cuda12")
_fake_cuda_plugin.initialize = lambda: None
sys.modules["jax_plugins.xla_cuda12"] = _fake_cuda_plugin
