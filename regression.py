from functools import partial
import jax
import jax.numpy as jnp
from disk_utils_jax import jax_model, jax_model_1d


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