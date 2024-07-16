from astropy.io import fits
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from regression import log_likelihood_1d_full_opt
from SLD_utils import *
from disk_utils_jax import jax_model_all_1d_full
from optimize import quick_optimize, quick_image
from scipy.optimize import minimize
from datetime import datetime

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.20'
jax.config.update("jax_enable_x64", True)


# Creating error map
def create_circular_err_map(image_shape, iradius, oradius, noise_level):
    err_map = jnp.zeros(image_shape)
    center = image_shape[0]/2
    y, x = jnp.ogrid[:image_shape[0], :image_shape[1]]
    distance = jnp.sqrt((x - center) ** 2 + (y - center) ** 2)  
    err_map = jnp.where(distance <= oradius, noise_level, 0)
    err_map = jnp.where(distance >= iradius, err_map, 0)
    return err_map

def process_image(image, scale_factor=1, offset=1):
    scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
    cropped_image = image[70:210, 70:210]
    def safe_float32_conversion(value):
        try:
            return np.float32(value)
        except (ValueError, TypeError):
            print("This value is unjaxable: " + str(value))
    fin_image = np.nan_to_num(cropped_image)
    fin_image = np.vectorize(safe_float32_conversion)(fin_image)
    return fin_image

def get_inc_bounded_knots(inclination, num_knots, buffer = 0):
    return jnp.linspace(jnp.cos(jnp.deg2rad(inclination-buffer)), jnp.cos(jnp.deg2rad(180-inclination+buffer)), num_knots)


hdul = fits.open("Fits/hr4796a_H_pol.fits")
target_image = process_image(hdul['SCI'].data[1,:,:])

from optimize import quick_optimize_full_opt, quick_image_full_opt

jax.config.update("jax_debug_nans", True)

err_map = create_circular_err_map(target_image.shape, 12, 83, 10)

knots = get_inc_bounded_knots(76.5, 5, buffer = 0)
init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
init_disk_guess = jnp.array([5., -5., 78.5, 76.5, 26.1])
init_cent_guess = jnp.array([70., 70.])
distr_guess = jnp.array([0, 3, 2, 1])
init_guess = jnp.concatenate([init_disk_guess, init_cent_guess, distr_guess, init_knot_guess])

init_image = quick_image_full_opt(init_guess, PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = 72.78, knots=knots)

llp = lambda x: log_likelihood_1d_full_opt(x, 
                    DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                    1e6, target_image, err_map, PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = 72.78,
                    knots=knots)


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: e, 8: ksi, 9: gamma, 10: beta
start = datetime.now()
soln = quick_optimize_full_opt(target_image, err_map, init_params=init_guess, method = None, iters = 1000, PSFModel=EMP_PSF, pxInArcsec=0.01414, distance = 72.78, knots = knots)
end = datetime.now()
print(soln)
print(end-start)

cent_image = quick_image_full_opt(soln, PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = 72.78, knots=knots)

fig, axes = plt.subplots(2,2, figsize=(20,10))

im = axes[0][0].imshow(target_image, origin='lower', cmap='inferno')
axes[0][0].set_title("Original Image")
plt.colorbar(im, ax=axes[0][0], shrink=0.75)

im = axes[0][1].imshow(init_image, origin='lower', cmap='inferno')
axes[0][1].set_title("Initial Guess")
plt.colorbar(im, ax=axes[0][1], shrink=0.75)

im = axes[1][0].imshow(cent_image, origin='lower', cmap='inferno')
axes[1][0].set_title("Fitted Disk")
plt.colorbar(im, ax=axes[1][0], shrink=0.75)

im = axes[1][1].imshow(target_image-cent_image, origin='lower', cmap='inferno')
axes[1][1].set_title("Final Image - Fitted Disk")
plt.colorbar(im, ax=axes[1][1], shrink=0.75)