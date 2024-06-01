from SLD_utils import *
import numpy as np
from regression import * 


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

disk_params1 = {}
disk_params1['inclination'] = 75. #In degrees
disk_params1['position_angle'] = 50. #In Degrees
disk_params1['alpha_in'] = 8. #The inner power law
disk_params1['alpha_out'] = -7. #The outer power law
#gs_ws = jnp.array([0.8,-0.2,0,0.75,0.25,0.]) #Here we have 3 Henyey-Greenstein functions with g parameters of 1, -1, and 0. The weights are 0.75, 0.25, and 0 respectively. 
disk_params1['flux_scaling'] = 1e6

#The disk size
disk_params1['sma'] = 30. #This is the semi-major axis of the model in astronomical units. 
#To get this in pixels, divide by the distance to the star, to get it in arcseconds. To get it in pixeks, divide by the pixel scale.
spf_params={'g1': 0.0, 'g2': -0.2, 'weight': 0.7}

disk_params_1d = np.array([disk_params1['alpha_in'], disk_params1['alpha_out'], disk_params1['sma'], disk_params1['inclination'],
                           disk_params1['position_angle']])

print(disk_params_1d)

from disk_utils_jax import jax_model
disk_image1 = jax_model(DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, disk_params=disk_params1,
                             spf_params = spf_params)

noise_level = 200
noise = np.random.normal(0, noise_level, disk_image1.shape)


test = log_likelihood_1d(disk_params_1d, DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, spf_params, disk_params1['flux_scaling'],
                    disk_image1+noise, jnp.ones(disk_image1.shape)*noise_level)


grad_func = jax.grad(log_likelihood_1d, argnums=0)

test = grad_func(disk_params_1d, DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, 
          spf_params, disk_params1['flux_scaling'], disk_image1+noise, 
          jnp.ones(disk_image1.shape)*noise_level)

print(test)

