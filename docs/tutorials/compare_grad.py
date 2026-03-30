"""Compare scipy_bounded_optimize with use_grad=True vs False."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF, recommended_num_knots,
)
from grater_jax.disk_model.objective_functions import Parameter_Index
from grater_jax.optimization.optimize_framework import Optimizer

G_TRUE = 0.3
hg_at_90 = float((1.0/(4.0*np.pi)) * (1 - G_TRUE**2) / (1 + G_TRUE**2)**1.5)
NX, NY    = 140, 140
FLUX      = 1e6
NUM_KNOTS = 6
INIT_G    = 0.5
KNOT_BOUND_BUFFER = 0.1
INCLINATIONS = [30.0, 50.0, 70.0, 80.0]

BASE_MISC = Parameter_Index.misc_params.copy()
BASE_MISC.update({"nx": NX, "ny": NY, "distance": 50.0, "pxInArcsec": 0.01225,
                  "halfNbSlices": 25, "flux_scaling": FLUX})
BASE_DISK = Parameter_Index.disk_params.copy()
BASE_DISK.update({"sma": 46.0, "alpha_in": 5.0, "alpha_out": -5.0, "ksi0": 1.0,
                  "gamma": 2.0, "beta": 1.0, "e": 0.0, "dens_at_r0": 1.0,
                  "position_angle": 0.0, "omega": 0.0,
                  "x_center": NX/2, "y_center": NY/2, "halfNbSlices": 25})
HG_SPF_PARAMS = HenyeyGreenstein_SPF.params.copy()
HG_SPF_PARAMS["g"] = G_TRUE

def hg_normalized(cos_phi, g):
    raw  = (1.0/(4.0*np.pi)) * (1-g**2) / (1+g**2-2*g*cos_phi)**1.5
    norm = (1.0/(4.0*np.pi)) * (1-g**2) / (1+g**2)**1.5
    return raw / norm

def probed_cosphi_range(incl_deg):
    s = np.sin(np.radians(incl_deg))
    return -s, s

cos_grid = np.linspace(-1.0, 1.0, 500)

print(f"{'incl':>5} | {'mode':>8} | {'nit':>5} | {'nfev':>5} | {'LL':>10} | {'max_err%':>9} | {'time(s)':>8} | success")
print("-" * 80)

for incl in INCLINATIONS:
    cp_back, cp_fwd = probed_cosphi_range(incl)
    nk = recommended_num_knots(incl, NUM_KNOTS, boundary_buffer=KNOT_BOUND_BUFFER)
    fwd_bound  = float(np.clip(cp_fwd  + KNOT_BOUND_BUFFER, -1.0, 1.0))
    back_bound = float(np.clip(cp_back - KNOT_BOUND_BUFFER, -1.0, 1.0))

    disk_params = BASE_DISK.copy(); disk_params["inclination"] = incl
    truth_misc  = BASE_MISC.copy(); truth_misc["flux_scaling"] = FLUX
    truth_opt   = Optimizer(ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                            HenyeyGreenstein_SPF, None,
                            disk_params, HG_SPF_PARAMS, None, truth_misc)
    truth_img   = np.array(truth_opt.get_model())
    err_map     = np.sqrt(np.abs(truth_img)) * np.sqrt(truth_img.max()) * 0.01
    in_range    = (cos_grid >= cp_back) & (cos_grid <= cp_fwd)
    true_spf    = hg_normalized(cos_grid, G_TRUE)

    for use_grad in [True, False]:
        spf_params = InterpolatedUnivariateSpline_SPF.params.copy()
        spf_params.update({"num_knots": nk, "knot_values": jnp.ones(nk),
                            "forwardscatt_bound": fwd_bound, "backscatt_bound": back_bound})
        fit_misc = BASE_MISC.copy(); fit_misc["flux_scaling"] = FLUX * hg_at_90
        opt = Optimizer(ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                        InterpolatedUnivariateSpline_SPF, None,
                        disk_params, spf_params, None, fit_misc)

        knot_cos      = np.array(InterpolatedUnivariateSpline_SPF.get_knots(opt.spf_params))
        n_left_init   = nk // 2
        free_knot_cos = np.concatenate([knot_cos[:n_left_init], knot_cos[n_left_init+1:]])
        opt.spf_params["knot_values"]   = jnp.array(hg_normalized(free_knot_cos, INIT_G))
        opt.misc_params["flux_scaling"] = FLUX * hg_at_90
        joint_bounds = ([np.full(nk, 1e-6), 1e2], [np.full(nk, 1e3), 1e12])

        t0 = time.perf_counter()
        soln = opt.scipy_bounded_optimize(
            fit_keys=["knot_values", "flux_scaling"], fit_bounds=joint_bounds,
            logscaled_params=["knot_values", "flux_scaling"], array_params=["knot_values"],
            target_image=truth_img, err_map=err_map,
            use_grad=use_grad, iters=2000, ftol=1e-14, gtol=1e-14,
        )
        elapsed = time.perf_counter() - t0

        ll = opt.log_likelihood(truth_img, err_map)
        spline_model = InterpolatedUnivariateSpline_SPF.pack_pars(
            opt.spf_params["knot_values"],
            knots=InterpolatedUnivariateSpline_SPF.get_knots(opt.spf_params),
        )
        recovered = np.array(InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            spline_model, jnp.array(cos_grid)
        ))
        max_err = float(np.max(
            np.abs(recovered[in_range] - true_spf[in_range]) / (np.abs(true_spf[in_range]) + 1e-30)
        )) * 100

        label = "grad" if use_grad else "no-grad"
        print(f"{incl:>5.0f} | {label:>8} | {soln.nit:>5} | {soln.nfev:>5} | "
              f"{ll:>10.4f} | {max_err:>9.2f} | {elapsed:>8.1f} | {soln.success}", flush=True)
