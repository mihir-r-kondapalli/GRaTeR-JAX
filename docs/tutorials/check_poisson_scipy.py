"""
check_poisson_scipy.py
======================
Quick sanity check: run scipy SPF recovery for the two high-inclination cases
(70° and 80°) under the new Poisson-like error map, and plot the results.

Run from the repo root:
    python docs/tutorials/check_poisson_scipy.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    HenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF,
    recommended_num_knots,
)
from grater_jax.disk_model.objective_functions import Parameter_Index
from grater_jax.optimization.optimize_framework import Optimizer

# ── Constants (mirror notebook) ───────────────────────────────────────────────
G_TRUE    = 0.3
hg_at_90  = float((1.0 / (4.0 * np.pi)) * (1 - G_TRUE**2) / (1 + G_TRUE**2)**1.5)

NX, NY    = 140, 140
FLUX      = 1e6
NUM_KNOTS = 6
FIT_ITERS = 2000
INIT_G    = 0.5
KNOT_BOUND_BUFFER = 0.1

INCLINATIONS = [70.0, 80.0]
PLOT_DIR = os.path.join(os.path.dirname(__file__), "spline_recovery_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

BASE_MISC = Parameter_Index.misc_params.copy()
BASE_MISC.update({
    "nx": NX, "ny": NY,
    "distance": 50.0,
    "pxInArcsec": 0.01225,
    "halfNbSlices": 25,
    "flux_scaling": FLUX,
})

BASE_DISK = Parameter_Index.disk_params.copy()
BASE_DISK.update({
    "sma": 46.0, "alpha_in": 5.0, "alpha_out": -5.0,
    "ksi0": 1.0, "gamma": 2.0, "beta": 1.0,
    "e": 0.0, "dens_at_r0": 1.0,
    "position_angle": 0.0, "omega": 0.0,
    "x_center": NX / 2, "y_center": NY / 2,
    "halfNbSlices": 25,
})

HG_SPF_PARAMS = HenyeyGreenstein_SPF.params.copy()
HG_SPF_PARAMS["g"] = G_TRUE

# ── Helpers ───────────────────────────────────────────────────────────────────
def hg_normalized(cos_phi, g):
    raw  = (1.0 / (4.0 * np.pi)) * (1 - g**2) / (1 + g**2 - 2*g*cos_phi)**1.5
    norm = (1.0 / (4.0 * np.pi)) * (1 - g**2) / (1 + g**2)**1.5
    return raw / norm

def probed_cosphi_range(incl_deg):
    s = np.sin(np.radians(incl_deg))
    return -s, s  # (cp_back, cp_fwd)

cos_grid = np.linspace(-1.0, 1.0, 500)
angles   = np.degrees(np.arccos(cos_grid))

JOINT_KEYS      = ["knot_values", "flux_scaling"]
JOINT_LOGSCALED = ["knot_values", "flux_scaling"]
JOINT_ARRAYS    = ["knot_values"]

# ── Main loop ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SPF recovery — Poisson error map  (scipy only)", fontsize=12)

for ax, incl in zip(axes, INCLINATIONS):
    print(f"\n{'='*55}\nInclination = {incl} deg")

    cp_back, cp_fwd = probed_cosphi_range(incl)
    nk         = recommended_num_knots(incl, NUM_KNOTS, boundary_buffer=KNOT_BOUND_BUFFER)
    fwd_bound  = float(np.clip(cp_fwd  + KNOT_BOUND_BUFFER, -1.0, 1.0))
    back_bound = float(np.clip(cp_back - KNOT_BOUND_BUFFER, -1.0, 1.0))

    # Build truth image
    disk_params = BASE_DISK.copy()
    disk_params["inclination"] = incl

    truth_misc = BASE_MISC.copy()
    truth_misc["flux_scaling"] = FLUX

    truth_opt = Optimizer(
        ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
        HenyeyGreenstein_SPF, None,
        disk_params, HG_SPF_PARAMS, None, truth_misc,
    )
    truth_img = np.array(truth_opt.get_model())

    # Poisson-like error map: sigma proportional to sqrt(flux),
    # scaled so peak SNR matches the old uniform 1% map.
    err_map = np.sqrt(np.abs(truth_img)) * np.sqrt(truth_img.max()) * 0.01
    print(f"  err_map: min={err_map.min():.4e}  max={err_map.max():.4e}")

    # Set up spline optimizer
    spf_params = InterpolatedUnivariateSpline_SPF.params.copy()
    spf_params["num_knots"]          = nk
    spf_params["knot_values"]        = jnp.ones(nk)
    spf_params["forwardscatt_bound"] = fwd_bound
    spf_params["backscatt_bound"]    = back_bound

    fit_misc = BASE_MISC.copy()
    fit_misc["flux_scaling"] = FLUX * hg_at_90

    opt = Optimizer(
        ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
        InterpolatedUnivariateSpline_SPF, None,
        disk_params, spf_params, None, fit_misc,
    )

    knot_cos = np.array(InterpolatedUnivariateSpline_SPF.get_knots(opt.spf_params))
    n_left_init   = nk // 2
    free_knot_cos = np.concatenate([knot_cos[:n_left_init], knot_cos[n_left_init + 1:]])
    init_kv = hg_normalized(free_knot_cos, INIT_G)
    opt.spf_params["knot_values"]   = jnp.array(init_kv)
    opt.misc_params["flux_scaling"] = FLUX * hg_at_90
    joint_bounds = ([np.full(nk, 1e-6), 1e2], [np.full(nk, 1e3), 1e12])

    nit_total = 0
    for pass_num in range(1, 4):
        soln = opt.scipy_bounded_optimize(
            fit_keys=JOINT_KEYS,
            fit_bounds=joint_bounds,
            logscaled_params=JOINT_LOGSCALED,
            array_params=JOINT_ARRAYS,
            target_image=truth_img,
            err_map=err_map,
            use_grad=True, iters=FIT_ITERS, ftol=1e-14, gtol=1e-14,
        )
        nit_total += soln.nit
        status = "converged" if soln.success else soln.message[:40].strip()
        print(f"  Pass {pass_num}: nit={soln.nit}  LL={opt.log_likelihood(truth_img, err_map):.4f}  [{status}]")
        if soln.success:
            break

    # Evaluate recovered SPF
    spline_model = InterpolatedUnivariateSpline_SPF.pack_pars(
        opt.spf_params["knot_values"],
        knots=InterpolatedUnivariateSpline_SPF.get_knots(opt.spf_params),
    )
    recovered_spf = np.array(
        InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            spline_model, jnp.array(cos_grid)
        )
    )
    true_spf_norm = hg_normalized(cos_grid, G_TRUE)

    in_range = (cos_grid >= cp_back) & (cos_grid <= cp_fwd)
    frac_err = (np.abs(recovered_spf[in_range] - true_spf_norm[in_range])
                / (np.abs(true_spf_norm[in_range]) + 1e-30))
    max_frac_err = float(frac_err.max())
    print(f"  Max SPF error (probed range): {max_frac_err*100:.2f}%")

    # Plot
    ang_fwd  = float(np.degrees(np.arccos(cp_fwd)))
    ang_back = float(np.degrees(np.arccos(cp_back)))
    ax.plot(angles, true_spf_norm, "k-", lw=2, label="True HG")
    ax.plot(angles, recovered_spf, "C0-", lw=2, label=f"Recovered (err={max_frac_err*100:.1f}%)")
    ax.axvspan(ang_fwd, ang_back, alpha=0.12, color="C0",
               label=f"Probed ({ang_fwd:.0f}–{ang_back:.0f}°)")
    ax.axvline(90, color="grey", ls="--", lw=0.8)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Scattering angle (deg)")
    ax.set_ylabel("Normalised SPF")
    ax.set_title(f"incl={incl}°  nk={nk}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 180)

out_path = os.path.join(PLOT_DIR, "poisson_scipy_check.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
