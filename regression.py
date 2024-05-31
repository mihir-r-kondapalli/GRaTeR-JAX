import jax
import jax.numpy as jnp
import numpy as np
from disk_utils_jax import jax_model, jax_model_1d
from functools import partial


@partial(jax.jit, static_argnums=(0,1))
def likelihood(DistrModel, FuncModel, disk_params, spf_params, target_image, err_map=jnp.ones([49, 140, 140])):
    model_image = jax_model(DistrModel, FuncModel, disk_params=disk_params, spf_params=spf_params) # (y)
    log_target = jnp.log(target_image)
    yerr = jnp.power((model_image-target_image)/(err_map),2)
    sigma2 = jnp.power(yerr, 2) + jnp.power(model_image, 2) * jnp.exp(2 * log_target)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

@partial(jax.jit, static_argnums=(1,2))
def likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map=jnp.ones([49, 140, 140])):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling) # (y)
    log_target = jnp.log(target_image)
    yerr = jnp.power((model_image-target_image)/(err_map),2)
    sigma2 = jnp.power(yerr, 2) + jnp.power(model_image, 2) * jnp.exp(2 * log_target)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Brute Force Regression Method
def manual_regression(target_image, DistrModel, FuncModel, spf_params, flux_scaling=1e6, bounds=None):
    # 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle
    disk_params = np.array([3, -3, 20, 20, 5])

    min_val = -likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image)

    min_ain, min_aout, min_sma, min_inc, min_pa = 3, -3, 20, 20, 5

    for i in range(min_ain, 10):
        disk_params[0] = float(i)

        for j in range(min_aout, -10, -1):
            disk_params[1] = float(j)

            for k in range(min_sma, 80):
                disk_params[2] = float(k)

                for l in range(min_inc, 70):
                    disk_params[3] = float(l)

                    for m in range(min_pa, 20):
                        disk_params[4] = float(m)

                        val = -likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image)

                        if(val < min_val):
                            min_ain = i
                            min_aout = j
                            min_sma = k
                            min_inc = l
                            min_pa = m
                            min_val = val

    disk_params[0] = min_ain
    disk_params[1] = min_aout
    disk_params[2] = min_sma
    disk_params[3] = min_inc
    disk_params[4] = min_pa

    #return {'alpha_in': min_ain, 'alpha_out': min_aout, 'sma': min_sma, 'inclination': min_inc, 'position_angle': min_pa, 'flux_scaling': flux_scaling}
    return {'alpha_in': min_ain, 'alpha_out': min_aout, 'sma': min_sma, 'inclination': min_inc, 'position_angle': min_pa, 'flux_scaling': flux_scaling}