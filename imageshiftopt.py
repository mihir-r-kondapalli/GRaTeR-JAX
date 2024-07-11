from astropy.io import fits
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from regression import log_likelihood_1d_pos_all_pars_spline
from SLD_utils import *
from disk_utils_jax import jax_model_1d
from optimize import quick_optimize, quick_image
from scipy.optimize import minimize

# Creating error map
def create_circular_err_map(image_shape, iradius, oradius, noise_level):
    err_map = jnp.zeros(image_shape)
    center = image_shape[0]/2
    y, x = jnp.ogrid[:image_shape[0], :image_shape[1]]
    distance = jnp.sqrt((x - center) ** 2 + (y - center) ** 2)  
    err_map = jnp.where(distance <= oradius, noise_level, 0)
    err_map = jnp.where(distance >= iradius, err_map, 0)
    return err_map


@jax.jit
def shift_center(img, center, new_center=None, flipx=False, astr_hdr=None):
    #create the coordinate system of the image to manipulate for the transform
    dims = img.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))

    #if necessary, move coordinates to new center
    if new_center is not None:
        dx = new_center[0] - center[0]
        dy = new_center[1] - center[1]
        x -= dx
        y -= dy

    #flip x if needed to get East left of North
    if flipx is True:
        x = center[0] - (x - center[0])

    # resampled_img = jax_pyklip_nan_map_coordinates_2d(img, yp, xp)
    resampled_img = jax.scipy.ndimage.map_coordinates(jnp.copy(img), jnp.array([y, x]),order=1,cval = 0.)

    return resampled_img

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

hdul = fits.open("Fits/hr4796a_H_pol.fits")
target_image = process_image(hdul['SCI'].data[1,:,:])


# Parameters
disk_params = {}
disk_params['inclination'] = 40. #In degrees
disk_params['position_angle'] = 50. #In Degrees
disk_params['alpha_in'] = 8. #The inner power law
disk_params['alpha_out'] = -5. #The outer power law
disk_params['flux_scaling'] = 1e6
disk_params['sma'] = 30. #This is the semi-major axis of the model in astronomical units. 
#To get this in pixels, divide by the distance to the star, to get it in arcseconds. To get it in pixeks, divide by the pixel scale.

disk_params_1d = np.array([disk_params['alpha_in'], disk_params['alpha_out'], disk_params['sma'], disk_params['inclination'],
                           disk_params['position_angle']])
spline_params_1d= jnp.full(6, 0.05)   # random knot y-values
all_pars_spline = jnp.concatenate([disk_params_1d, spline_params_1d])

err_map = create_circular_err_map(target_image.shape, 12, 83, 5)


ll = lambda x, y: log_likelihood_1d_pos_all_pars_spline(x, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        disk_params['flux_scaling'], y, err_map, PSFModel=GAUSSIAN_PSF, pxInArcsec=0.01414)

def likelihood_image_shift(pars):
    return ll(pars[2:], shift_center(target_image, (70,70), new_center=(pars[0], pars[1])))

all_pars_cent = jnp.concatenate([jnp.array([70, 70]), all_pars_spline])

print("hi")
min_pars = minimize(likelihood_image_shift, all_pars_cent)
print("hi")
print(min_pars)
cent_image = shift_center(quick_image(min_pars.x[2:], PSFModel=GAUSSIAN_PSF), (70,70), new_center=(min_pars.x[0], min_pars.x[1]))

fig, axes = plt.subplots(1,3, figsize=(20,10))

im = axes[0].imshow(target_image, origin='lower', cmap='inferno')
axes[0].set_title("Original Image")
plt.colorbar(im, ax=axes[0], shrink=0.75)

im = axes[1].imshow(cent_image, origin='lower', cmap='inferno')
axes[1].set_title("Center Image")
plt.colorbar(im, ax=axes[1], shrink=0.75)

im = axes[2].imshow(target_image-cent_image, origin='lower', cmap='inferno')
axes[2].set_title("Final Image - Center Image")
plt.colorbar(im, ax=axes[2], shrink=0.75)


fig.savefig('output.png')