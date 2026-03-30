"""
make_corner_plots.py
====================
Generate corner plots for the MCMC SPF knot posteriors from saved emcee backends.

Run from docs/tutorials/:
    python make_corner_plots.py
"""

import os
import sys

import numpy as np
import emcee
import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from grater_jax.disk_model.SLD_utils import InterpolatedUnivariateSpline_SPF, recommended_num_knots

G_TRUE            = 0.3
NUM_KNOTS         = 6
KNOT_BOUND_BUFFER = 0.1
INCLINATIONS      = [30.0, 50.0, 70.0, 80.0]
MCMC_DIR          = "mcmc_backends"
PLOT_DIR          = "spline_recovery_plots"
BURNS             = 200
SUBSAMPLE         = 20000
RNG               = np.random.default_rng(42)


def hg_normalized(cos_phi, g):
    raw  = (1.0 / (4.0 * np.pi)) * (1 - g**2) / (1 + g**2 - 2*g*cos_phi)**1.5
    norm = (1.0 / (4.0 * np.pi)) * (1 - g**2) / (1 + g**2)**1.5
    return raw / norm


def probed_cosphi_range(incl_deg):
    s = np.sin(np.radians(incl_deg))
    return -s, s


for incl in INCLINATIONS:
    print(f"\nincl = {incl:.0f}°", flush=True)
    nk = recommended_num_knots(incl, NUM_KNOTS, boundary_buffer=KNOT_BOUND_BUFFER)
    cp_back, cp_fwd = probed_cosphi_range(incl)
    fwd_bound  = float(np.clip(cp_fwd  + KNOT_BOUND_BUFFER, -1.0, 1.0))
    back_bound = float(np.clip(cp_back - KNOT_BOUND_BUFFER, -1.0, 1.0))

    spf_params = InterpolatedUnivariateSpline_SPF.params.copy()
    spf_params.update({
        "num_knots": nk, "knot_values": jnp.ones(nk),
        "forwardscatt_bound": fwd_bound, "backscatt_bound": back_bound,
    })
    # get_knots returns nk+1 positions; the center knot (index nk//2) is fixed
    # at value 1.0 and not sampled — drop it to match the chain's nk columns
    all_knots   = np.array(InterpolatedUnivariateSpline_SPF.get_knots(spf_params))
    n_left      = nk // 2
    knots       = np.concatenate([all_knots[:n_left], all_knots[n_left + 1:]])
    knot_angles = np.degrees(np.arccos(knots))

    backend_path = os.path.join(MCMC_DIR, f"spline_incl{int(incl):02d}_emcee_backend.h5")
    reader    = emcee.backends.HDFBackend(backend_path, read_only=True)
    flatchain = reader.get_chain(discard=BURNS, flat=True)
    print(f"  flatchain: {flatchain.shape}", flush=True)

    # knot_values only (drop flux_scaling last column); un-log-scale
    kv_phys = np.exp(flatchain[:, :nk])
    idx     = RNG.choice(len(kv_phys), size=min(SUBSAMPLE, len(kv_phys)), replace=False)
    kv_sub  = kv_phys[idx]

    true_kv = hg_normalized(knots, G_TRUE)
    labels  = [f"{a:.0f}°" for a in knot_angles]

    print("  building corner plot...", flush=True)
    fig = corner.corner(
        kv_sub,
        labels=labels,
        truths=true_kv,
        truth_color="k",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        title_fmt=".3f",
        color="C0",
    )
    fig.suptitle(f"SPF knot posterior  ($i = {incl:.0f}$°)", fontsize=12, y=1.01)

    out_png = os.path.join(PLOT_DIR, f"mcmc_corner_incl{int(incl):02d}.png")
    out_pdf = os.path.join(PLOT_DIR, f"mcmc_corner_incl{int(incl):02d}.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}  {out_pdf}", flush=True)

print("\nDone.")
