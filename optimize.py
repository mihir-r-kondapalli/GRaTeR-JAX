from regression import log_likelihood_1d_pos_all_pars_spline, log_likelihood_1d_full_opt, log_likelihood_1d_pos_cent
from disk_utils_jax import jax_model_all_1d, jax_model_all_1d_cent, jax_model_all_1d_full
from scipy.optimize import minimize
from SLD_utils import *
import jax.numpy as jnp


FULL_BOUNDS = [(1, 10), (-1, 10), (10, 80), (0, 180), (0, 180), (60, 80), (60, 80), (0, 10), (0, 10), (0, 10),
              (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)]

def quick_optimize(target_image, err_map, flux_scaling=1e6, knots=jnp.linspace(1, -1, 6), grad = False, disp = True, method = None,
                    iters = 500, **kwargs):
    init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
    init_disk_guess = jnp.array([5., -5., 45., 45, 45])
    init_guess = jnp.concatenate([init_disk_guess, init_knot_guess])
    llp = lambda x: log_likelihood_1d_pos_all_pars_spline(x, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, **kwargs)
    grad_func = None
    if(grad):
        grad_func = jax.grad(llp)

    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, jac=grad_func)
    if(disp):
        print(soln)
    return soln.x

def quick_image(pars, flux_scaling=1e6, **kwargs):
    return jax_model_all_1d(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0:5],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[5:]), flux_scaling, **kwargs)

# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc
# 7 onwards is spline parameters, pxInArcsec and distance are good kwargs to include
def quick_optimize_cent(target_image, err_map, flux_scaling=1e6, knots=jnp.linspace(1, -1, 6), disp = True, method = None,
                    iters = 500, bounds = None, **kwargs):

    init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
    init_disk_guess = jnp.array([5., -5., 45., 45, 45])
    init_cent_guess = jnp.array([70., 70.])
    init_guess = jnp.concatenate([init_cent_guess, init_disk_guess, init_knot_guess])
    llp = lambda x: log_likelihood_1d_pos_cent(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, **kwargs)
    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, bounds=bounds,)
    if(disp):
        print(soln)
    return soln.x

@partial(jax.jit, static_argnums=(2))
def quick_image_cent(pars, flux_scaling=1e6, PSFModel = None, **kwargs):
    return jax_model_all_1d_cent(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0], pars[1], pars[2:7],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[7:]), flux_scaling, PSFModel=PSFModel, **kwargs)


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: ksi, 8: gamma, 9: beta
# 11 onwards is spline parameters, pxInArcsec and distance are good kwargs to include
def quick_optimize_full_opt(target_image, err_map, flux_scaling=1e6, knots=jnp.linspace(1, -1, 6), disp = True, method = None,
                    iters = 500, bounds = None, **kwargs):

    init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
    init_disk_guess = jnp.array([5., -5., 45., 45, 45])
    init_cent_guess = jnp.array([70., 70.])
    distr_guess = jnp.array([0, 3, 2, 1])
    init_guess = jnp.concatenate([init_disk_guess, init_cent_guess, distr_guess, init_knot_guess])
    llp = lambda x: log_likelihood_1d_full_opt(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, **kwargs)

    print(llp(init_guess).shape)
    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, bounds=bounds)
    if(disp):
        print(soln)
    return soln.x

def quick_image_full_opt(pars, flux_scaling=1e6, **kwargs):
    return jax_model_all_1d_full(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0:11],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[11:]), flux_scaling, **kwargs)