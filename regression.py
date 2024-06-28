import jax
import jax.numpy as jnp
import numpy as np
from disk_utils_jax import jax_model, jax_model_1d, jax_model_all_1d
from functools import partial


# Computes the error between the model_image and target_image
@jax.jit
def log_likelihood_image(model_image, target_image, err_map):
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters
@partial(jax.jit, static_argnums=(0,1))
def log_likelihood(DistrModel, FuncModel, disk_params, spf_params, target_image, err_map):
    model_image = jax_model(DistrModel, FuncModel, disk_params=disk_params, spf_params=spf_params) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_params is a jax array)
@partial(jax.jit, static_argnums=(1,2))
def log_likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_params is a jax array)
# Returns a positive number instead of negative number for future use
@partial(jax.jit, static_argnums=(1,2))
def log_likelihood_1d_pos(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_and_spf_params is a jax array
# and a combination of a disk_params array and an spf_param array) (first 6 values are disk's, rest are spf's)
# Returns a positive number instead of negative number for future use
# This method does not work with spline spfs
@partial(jax.jit, static_argnums=(1,2))
def log_likelihood_1d_pos_all_pars(disk_and_spf_params, DistrModel, FuncModel, flux_scaling, target_image, err_map):
    model_image = jax_model_all_1d(DistrModel, FuncModel, disk_and_spf_params[0:5], disk_and_spf_params[5:], flux_scaling) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_and_spf_params is a jax array
# and a combination of a disk_params array and an spf_param array) (first 6 values are disk's, rest are spf's)
# Returns a positive number instead of negative number for future use
# This method is exclusively for spline spfs
@partial(jax.jit, static_argnums=(1,2,6,7))
def log_likelihood_1d_pos_all_pars_spline(disk_and_spf_params, DistrModel, FuncModel, flux_scaling, target_image, err_map, inc = 0,
                                            knots = 10):
    model_image = jax_model_all_1d(DistrModel, FuncModel, disk_and_spf_params[0:5], FuncModel.pack_pars(disk_and_spf_params[5:],
                    inc=inc, knots=knots), flux_scaling) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)