"""
run_mcmc_poisson.py
===================
Full injection-recovery run (scipy + MCMC) for all inclinations using the
Poisson-like error map.  Mirrors the notebook logic exactly; run this instead
of nbconvert to avoid kernel-startup / CUDA-compilation timeout issues.

Run from docs/tutorials/:
    nohup python run_mcmc_poisson.py > run_mcmc_poisson.log 2>&1 &
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

from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    HenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF,
    recommended_num_knots,
)
from grater_jax.disk_model.objective_functions import Parameter_Index
from grater_jax.optimization.optimize_framework import Optimizer

# ── Constants ─────────────────────────────────────────────────────────────────
G_TRUE    = 0.3
hg_at_90  = float((1.0 / (4.0 * np.pi)) * (1 - G_TRUE**2) / (1 + G_TRUE**2)**1.5)

NX, NY    = 140, 140
FLUX      = 1e6
NUM_KNOTS = 6
FIT_ITERS = 2000
INIT_G    = 0.5
KNOT_BOUND_BUFFER = 0.1

INCLINATIONS = [30.0, 50.0, 70.0, 80.0]

NWALKERS = 64
NITER    = 2000
BURNS    = 200

PLOT_DIR = "spline_recovery_plots"
MCMC_DIR = "mcmc_backends"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MCMC_DIR, exist_ok=True)

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

JOINT_KEYS      = ["knot_values", "flux_scaling"]
JOINT_LOGSCALED = ["knot_values", "flux_scaling"]
JOINT_ARRAYS    = ["knot_values"]

MCMC_KEYS      = ["knot_values", "flux_scaling"]
MCMC_LOGSCALED = ["knot_values", "flux_scaling"]
MCMC_ARRAYS    = ["knot_values"]

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

# ── Scipy injection-recovery ──────────────────────────────────────────────────
print("=" * 60)
print("SCIPY INJECTION-RECOVERY")
print("=" * 60)

results = {}

for incl in INCLINATIONS:
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
    err_map   = np.sqrt(np.abs(truth_img)) * np.sqrt(truth_img.max()) * 0.01
    print(f"  Truth image max={truth_img.max():.4e}  err_map max={err_map.max():.4e}")

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

    knot_cos      = np.array(InterpolatedUnivariateSpline_SPF.get_knots(opt.spf_params))
    n_left_init   = nk // 2
    free_knot_cos = np.concatenate([knot_cos[:n_left_init], knot_cos[n_left_init + 1:]])
    opt.spf_params["knot_values"]   = jnp.array(hg_normalized(free_knot_cos, INIT_G))
    opt.misc_params["flux_scaling"] = FLUX * hg_at_90
    joint_bounds = ([np.full(nk, 1e-6), 1e2], [np.full(nk, 1e3), 1e12])

    nit_total = 0
    for pass_num in range(1, 4):
        soln = opt.scipy_bounded_optimize(
            fit_keys=JOINT_KEYS, fit_bounds=joint_bounds,
            logscaled_params=JOINT_LOGSCALED, array_params=JOINT_ARRAYS,
            target_image=truth_img, err_map=err_map,
            use_grad=True, iters=FIT_ITERS, ftol=1e-14, gtol=1e-14,
        )
        nit_total += soln.nit
        status = "converged" if soln.success else soln.message[:40].strip()
        print(f"  Pass {pass_num}: nit={soln.nit}  LL={opt.log_likelihood(truth_img, err_map):.4f}  [{status}]")
        if soln.success:
            break

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
    in_range  = (cos_grid >= cp_back) & (cos_grid <= cp_fwd)
    frac_err  = (np.abs(recovered_spf[in_range] - true_spf_norm[in_range])
                 / (np.abs(true_spf_norm[in_range]) + 1e-30))
    max_frac_err = float(frac_err.max())
    print(f"  Max SPF error (probed range): {max_frac_err*100:.2f}%")

    results[incl] = dict(
        truth_img=truth_img, err_map=err_map,
        recovered_spf=recovered_spf, true_spf_norm=true_spf_norm,
        knot_cos=knot_cos,
        knot_values=np.array(opt.spf_params["knot_values"]),
        flux_scaling=float(opt.misc_params["flux_scaling"]),
        spline_model=spline_model,
        cp_fwd=cp_fwd, cp_back=cp_back,
        fwd_bound=fwd_bound, back_bound=back_bound,
        in_range=in_range, max_frac_err=max_frac_err,
    )

# ── MCMC ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MCMC")
print("=" * 60)

mcmc_results = {}

for incl in INCLINATIONS:
    print(f"\n{'='*55}\nInclination = {incl} deg  [MCMC]")
    r  = results[incl]
    nk = recommended_num_knots(incl, NUM_KNOTS)

    fit_disk = BASE_DISK.copy()
    fit_disk["inclination"] = incl

    spf_params = InterpolatedUnivariateSpline_SPF.params.copy()
    spf_params["num_knots"]          = nk
    spf_params["knot_values"]        = jnp.array(r["knot_values"])
    spf_params["forwardscatt_bound"] = r["fwd_bound"]
    spf_params["backscatt_bound"]    = r["back_bound"]

    fit_misc = BASE_MISC.copy()
    fit_misc["flux_scaling"] = r["flux_scaling"]

    opt = Optimizer(
        ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
        InterpolatedUnivariateSpline_SPF, None,
        fit_disk, spf_params, None, fit_misc,
    )
    opt.name = os.path.join(MCMC_DIR, f"spline_incl{int(incl):02d}")

    mcmc_bounds = (
        [np.full(nk, 1e-6), 1e2],
        [np.full(nk, 1e3),  1e12],
    )

    mc_model = opt.mcmc(
        fit_keys=MCMC_KEYS,
        logscaled_params=MCMC_LOGSCALED,
        array_params=MCMC_ARRAYS,
        target_image=r["truth_img"],
        err_map=r["err_map"],
        BOUNDS=mcmc_bounds,
        nwalkers=NWALKERS, niter=NITER, burns=BURNS,
        confirm_overwrite=False,
    )
    print(f"  Done. Posterior samples: {mc_model._get_flatchain(scaled=True).shape[0]}")

    flatchain  = mc_model._get_flatchain(scaled=True)
    kv_samples = flatchain[:, :nk]
    knots      = InterpolatedUnivariateSpline_SPF.get_knots(spf_params)

    stride = max(1, len(kv_samples) // 2000)
    spf_samples = []
    for kv in kv_samples[::stride]:
        spline = InterpolatedUnivariateSpline_SPF.pack_pars(jnp.array(kv), knots=knots)
        curve  = np.array(InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            spline, jnp.array(cos_grid)
        ))
        spf_samples.append(curve)
    spf_samples = np.array(spf_samples)

    spf_median = np.median(spf_samples, axis=0)
    spf_lo     = np.percentile(spf_samples, 16, axis=0)
    spf_hi     = np.percentile(spf_samples, 84, axis=0)

    in_range = r["in_range"]
    frac_err_median = float(np.max(
        np.abs(spf_median[in_range] - r["true_spf_norm"][in_range])
        / (np.abs(r["true_spf_norm"][in_range]) + 1e-30)
    ))
    print(f"  Max SPF error (median, probed range): {frac_err_median*100:.2f}%")

    mcmc_results[incl] = dict(
        spf_samples=spf_samples, spf_median=spf_median,
        spf_lo=spf_lo, spf_hi=spf_hi,
        true_spf_norm=r["true_spf_norm"],
        recovered_spf=r["recovered_spf"],
        cp_fwd=r["cp_fwd"], cp_back=r["cp_back"],
        max_frac_err_median=frac_err_median,
    )

# ── MCMC posterior summary plot ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, incl in zip(axes.flat, INCLINATIONS):
    r  = mcmc_results[incl]
    cp_back, cp_fwd = r["cp_back"], r["cp_fwd"]
    ang_fwd  = float(np.degrees(np.arccos(cp_fwd)))
    ang_back = float(np.degrees(np.arccos(cp_back)))

    ax.fill_between(angles, r["spf_lo"], r["spf_hi"], alpha=0.35, color="C0", label="Posterior ±1σ")
    ax.plot(angles, r["spf_median"],    "C0-",  lw=1.5, label="Posterior median")
    ax.plot(angles, r["true_spf_norm"], "k-",   lw=2,   label="True HG")
    ax.plot(angles, r["recovered_spf"], "--",   lw=1.5, color="C1", label="Scipy best-fit")
    ax.axvspan(0,        ang_fwd,  alpha=0.08, color="grey")
    ax.axvspan(ang_back, 180,      alpha=0.08, color="grey",
               label=f"Probed ({ang_fwd:.0f}–{ang_back:.0f}°)")
    ax.axvline(90, color="grey", ls="--", lw=0.8)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Scattering angle (°)")
    ax.set_ylabel("Normalised SPF")
    ax.set_title(f"$i = {incl:.0f}$°  (median error = {r['max_frac_err_median']*100:.2f}%)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 180)

plt.tight_layout()
out_png = os.path.join(PLOT_DIR, "mcmc_spf_posterior_summary.png")
out_pdf = os.path.join(PLOT_DIR, "mcmc_spf_posterior_summary.pdf")
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
