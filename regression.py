import jax
import jax.numpy as jnp
import numpy as np
from disk_utils_jax import jax_model, jax_model_1d
from functools import partial


@partial(jax.jit, static_argnums=(0,1))
def log_likelihood(DistrModel, FuncModel, disk_params, spf_params, target_image, err_map):
    model_image = jax_model(DistrModel, FuncModel, disk_params=disk_params, spf_params=spf_params) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

@partial(jax.jit, static_argnums=(1,2))
def log_likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)