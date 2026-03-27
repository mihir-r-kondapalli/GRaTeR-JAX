# CLAUDE.md -- grater-jax

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRaTeR-JAX is a JAX-accelerated Python package for modeling scattered-light observations of circumstellar debris disks. It implements the GRaTeR (Generalized Radial Transporter) framework using automatic differentiation (via JAX) to enable gradient-based optimization and Bayesian inference. All steps are differentiable JAX functions, enabling future gradient-based optimization.

## Engineering Values

Ordered by priority when values conflict:

1. **Correctness first** — code that doesn't work right is worthless regardless of cleanliness.
2. **Explicit over clever** — if it requires a comment to explain *what* it does, rewrite it.
3. **Edge cases matter** — handle more, not fewer.
4. **DRY is load-bearing** — repeated literals → constants; repeated logic → functions;
   repeated patterns → abstractions. Magic numbers, URLs, paths, thresholds belong in a
   constants section or config object.
5. **Well-tested code is non-negotiable** — every public function gets tests; every error
   path gets tests; edge cases get tests.
6. **"Engineered enough"** — not fragile/hacky, not prematurely abstracted. Simplest correct
   solution wins; easier to add abstraction later than remove it.

## Programming Principles

**YAGNI + KISS**: implement what is asked. No speculative features. The simplest correct
solution wins. If a domain-knowledgeable developer can't follow it in 60 seconds, simplify.

**SOLID (pragmatically)**: each function does one thing. Prefer composition over inheritance.
Inject dependencies — don't hardcode I/O, APIs, or file access. Keep interfaces narrow.

**Defensive programming**: validate inputs at system boundaries. Use guard clauses and early
returns. Fail fast and loudly — never silently swallow errors.

**Separation of concerns**: I/O separate from logic. Parsing separate from processing.
Config separate from code. Functions that compute should not also print or write files.

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

## Python Standards

- Type hints on ALL signatures and class attributes.
- Use `from __future__ import annotations` for forward references.
- Modern syntax: `X | Y` union types, walrus operator where clear.
- Dataclasses or Pydantic for structured data. No raw dicts for domain objects.
- `pathlib` for paths. `logging` instead of print for operational output. f-strings for formatting.
- Context managers for resource management.

### Code Structure

- Module-level docstring: WHAT and WHY (not HOW).
- Imports: stdlib → third-party → local, blank line separators.
- Constants below imports. Public API before private helpers.
- Functions ~25 lines guideline; up to 50 with justification. Split files at ~300 lines.

### Naming

- Names reveal intent: `parse_resonator_frequencies()` not `process_data()`.
- Booleans read as assertions: `is_valid`, `has_converged`, `should_retry`.
- Collections are plural. Naming is consistent across the codebase.
- Units in variable names or docstrings: `frequency_hz`, `wavelength_m`, `height_nm`.

### Functions

- Typed parameters and return values. Limit to 3–4 args; group related params into a dataclass.
- No flag parameters that change behavior — split into two functions.
- NumPy-style docstrings on all public functions: summary, Parameters, Returns, Raises.

### Error Handling

- Library code: raise specific named exceptions, never print.
- Custom exceptions for domain errors. Never bare `Exception`.
- Error messages: what was attempted, what went wrong, what to do about it.

### Comments

- Comments explain WHY, not WHAT. No commented-out code.
- TODOs include reason and context.

## Scientific Computing Standards

- Be explicit about units in variable names or docstrings (`wavelength_m`, `radius_nm`).
- Guard against floating-point edge cases: division by zero, NaN propagation, loss of
  precision in subtraction of similar values.
- Prefer JAX/NumPy vectorized operations over Python loops for array data.
- Document physical assumptions and reference papers/equations by name.
- Validate array shapes at function entry for non-trivial operations.
- **JAX (`jnp`)** for all operations that may be differentiated or jit-compiled.
- **NumPy (`np`)** for I/O and data that never needs grad flow.

## Testing Standards

- **pytest** as the default framework. Tests in `tests/` mirroring source structure.
- Test names describe behavior: `test_cell_loss_raises_on_mismatched_wavelength_count`.
- One assertion per test concept (multiple asserts fine if testing one logical thing).
- Use fixtures and `parametrize` for repetition. No test interdependence.
- Cover: happy path, edge cases, error paths, boundary conditions.
- Integration tests are separate from unit tests and clearly labeled.

### Test-After-Build Requirement

After completing each implementation task, Claude Code **must**:
1. Run the relevant test file
2. Confirm all tests pass before moving to the next task.
3. If tests fail, fix the implementation (not the tests) before proceeding.
4. Report the test summary (N passed, 0 failed) as part of task completion.
5. **Commit the passing work** following the Git Discipline section format.

## Git Discipline

Claude Code must make incremental git commits throughout every session. Commits are
the audit trail for the session — they make it easy to review what was done, roll back
a bad decision, and resume work in a future session.

### Commit Rules

- **Commit after every completed task** in the Immediate Tasks table before moving to the next.
- **Commit after every passing test run** — the commit message must reference the test result.
- **Never bundle unrelated changes** into one commit. One logical unit of work = one commit.
- **Never commit broken code.** If tests fail, fix first, then commit.
- **Always `git add -p`** (patch mode) rather than `git add .` — review what is being staged.

### Commit Message Format

```
<type>(<scope>): <short summary>

<optional body: what changed and why, not how>

Tests: N passed, 0 failed
```

**Types**: `feat` (new capability), `fix` (bug fix), `test` (tests only), `docs` (notebooks,
docstrings, README), `refactor` (no behavior change), `chore` (scaffolding, config, deps).

**Examples:**
```
feat(geometry): add SuperellipsePost dataclass and constructor helpers

Implements the unified shape family replacing separate Cylinder/Ring/Rectangle classes.
n_out/n_in exponents use softplus parameterization to keep values ≥ 2.

Tests: 14 passed, 0 failed
```
```
feat(storage): add LibraryStore protocol and ZarrStore backend

ZarrStore is the default backend. HDF5Store stub added for future swap.
Backend selected via SimulationConfig.store_backend.

Tests: 8 passed, 0 failed
```
```
docs(notebooks): add 01_fdtdx_transition.py Marimo tutorial
```

### Session Start and End

**At the start of every session**, run:
```bash
git status
git log --oneline -10
```
Report the current branch and last 3 commits before doing any work. This orients the
session relative to prior work.

**At the end of every session**, ensure:
- All completed work is committed (no unstaged changes to finished code)
- Any work-in-progress is either committed with a `wip:` prefix or stashed
- Run `git log --oneline -5` and report the session's commits as a summary

### Branch Strategy

- Work on branch `briley-claude` for solo development.
- If the user asks for exploratory or experimental work, create a branch:
  `git checkout -b experiment/<short-description>`
- Never force-push without explicit user instruction.