
from astropy.io import fits
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from SLD_utils import (DoubleHenyeyGreenstein_SPF, InterpolatedUnivariateSpline_SPF, DustEllipticalDistribution2PowerLaws,
                    EMP_PSF)
import warnings
from optimize import quick_image_cent
from regression import log_likelihood_1d_pos_cent
from datetime import datetime
from mcmc_model import MCMC_model
from tqdm import tqdm

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

jax.config.update("jax_enable_x64", True)

###############################################################################
# Helper Methods

def create_empirical_err_map(data, annulus_width=5, mask_rad=9, outlier_pixels=None):    
    y,x = np.indices(data.shape)
    y -= data.shape[0]//2
    x -= data.shape[1]//2 
    radii = np.sqrt(x**2 + y**2) 
    noise_array = np.zeros_like(data)
    for i in range(0, int(np.max(radii)//annulus_width) ): 
        indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
        noise_array[indices] = np.nanstd(data[indices])
    mask = radii <= mask_rad
    noise_array[mask] = 0

    if(outlier_pixels != None):
        for pixel in outlier_pixels:
            noise_array[pixel[0]][pixel[1]] = noise_array[pixel[0]][pixel[1]] * 1e6 

    return noise_array

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

def get_inc_bounded_knots(inclination, radius, buffer = 0, num_knots=-1):
    if(num_knots <= 0):
        if(radius < 50):
            num_knots = 4
        else:
            num_knots = 6
    return jnp.linspace(jnp.cos(jnp.deg2rad(90-inclination-buffer)), jnp.cos(jnp.deg2rad(90+inclination+buffer)), num_knots)

def transpose_even_to_odd_spline(knot_vals, knots, new_knots):
    x_vals = jnp.linspace(-1, 1, 100)
    spline = InterpolatedUnivariateSpline_SPF.init(knot_vals, knots=knots)
    y_vals = spline(x_vals)

    new_k_vals = []
    for nk in new_knots:
        new_k_vals.append(y_vals[int((nk*100 + 100)/2)])
    return jnp.array(new_k_vals)

def plot_mc_img(target_image, sc_image, mc_image, init_val, fin_val):

    ## Get a good scaling
    y, x = np.indices(target_image.shape)
    y -= 70
    x -= 70 
    rads = np.sqrt(x**2+y**2)
    mask = (rads > 12)

    vmin = np.nanpercentile(target_image[mask], 1)
    vmax = np.nanpercentile(target_image[mask], 99.9)

    fig, axes = plt.subplots(3,3, figsize=(20,10))

    im = axes[0][0].imshow(target_image, origin='lower', cmap='inferno')
    axes[0][0].set_title("Target Image")
    plt.colorbar(im, ax=axes[0][0], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[0][1].imshow(sc_image, origin='lower', cmap='inferno')
    axes[0][1].set_title("Scipy Fitted Image: " + str(init_val))
    plt.colorbar(im, ax=axes[0][1], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][0].imshow(mc_image, origin='lower', cmap='inferno')
    axes[1][0].set_title("MCMC Fitted Image: " + str(fin_val))
    plt.colorbar(im, ax=axes[0][2], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][1].imshow(target_image-sc_image, origin='lower', cmap='inferno')
    axes[1][1].set_title("Scipy Residual")
    plt.colorbar(im, ax=axes[1][1], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][2].imshow(target_image-mc_image, origin='lower', cmap='inferno')
    axes[1][2].set_title("MCMC Residual")
    plt.colorbar(im, ax=axes[1][2], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[2][1].imshow((target_image-sc_image)/err_map, origin='lower', cmap='inferno')
    axes[2][1].set_title("Scipy Residual / Noise")
    plt.colorbar(im, ax=axes[2][1], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[2][2].imshow((target_image-mc_image)/err_map, origin='lower', cmap='inferno')
    axes[2][2].set_title("MCMC Residual / Noise")
    plt.colorbar(im, ax=axes[2][2], shrink=0.75)
    im.set_clim(vmin, vmax)

def get_aic(pos_log_likelihood, num_params):
    return 2 * pos_log_likelihood + 2 * num_params

def get_bic(pos_log_likelihood, num_params):
    return 2 * pos_log_likelihood + np.log(140*140) * num_params

###############################################################################

def fit_spline(row, target_image, err_map, disp = False, flux_scaling=1e6):

    knots = get_inc_bounded_knots(row["Inclination"], row["Radius"], buffer = 0, num_knots=int(row["Knots"]))

    init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
    disk_pars = jnp.array([row["xc"], row["yc"], row["Alpha_In"], row["Alpha_Out"],
                        row["Radius"], row["Inclination"], row["Position Angle"]])

    bounds = []
    for i in range(0, int(row["Knots"])):
        bounds.append((0, 0.1))

    llp_spline = lambda x: log_likelihood_1d_pos_cent(jnp.concatenate([disk_pars, x]), 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF,
                        flux_scaling, target_image, err_map, knots=knots, distance = row["Distance"],
                        PSFModel = EMP_PSF)

    maxiters = 1000
    opt = {'disp':disp,'maxiter':maxiters}
    knot_vals = minimize(llp_spline, init_knot_guess, options = opt, method = 'l-bfgs-b', bounds=bounds).x

    if (int(row['Knots'])%2 == 0):
        knot_vals = transpose_even_to_odd_spline(knot_vals, knots,
                    get_inc_bounded_knots(row["Inclination"], row["Radius"], buffer = 0, num_knots=int(row["Knots"]+1)))

    fin_pars = jnp.concatenate([disk_pars, knot_vals])

    return fin_pars


def run_mcmc_ab(soln, target_image, err_map, row, name, nwalkers=250, niter=250, burns=70):

    knots = get_inc_bounded_knots(row["Inclination"], row["Radius"], buffer = 0, num_knots=jnp.size(soln[7:]))

    llp = lambda x: -log_likelihood_1d_pos_cent(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        1e6, target_image, err_map, PSFModel = EMP_PSF, pxInArcsec=0.01414,
                        distance = row["Distance"], knots=knots)


    DISK_BOUNDS = np.array([np.array([0.1, -15, 0, 0, 0]), np.array([15, -0.1, 150, 180, 400])])
    CENT_BOUNDS = np.array([np.array([65, 65]), np.array([75, 75])])
    SPLINE_BOUNDS = np.array([np.zeros(jnp.size(knots)), 0.1*np.ones(jnp.size(knots))])
    BOUNDS = np.array([np.concatenate([CENT_BOUNDS[0], DISK_BOUNDS[0], SPLINE_BOUNDS[0]]),
                        np.concatenate([CENT_BOUNDS[1], DISK_BOUNDS[1], SPLINE_BOUNDS[1]])])

    if not(np.all(soln > BOUNDS[0]) and np.all(soln < BOUNDS[1])):
        print()
        print("Intial params out of bounds for MCMC!!!")
        print(soln)
        print(BOUNDS)
        print(not(soln > BOUNDS[0] and soln < BOUNDS[1]))
        exit()

    mc_model = MCMC_model(llp, BOUNDS)
    mc_model.run(soln, nconst = 1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns)

    # Plotting Corner Plots
    labels = ['xc', 'yc', 'alpha_in', 'alpha_out', 'sma', 'inclination', 'position_angle']
    for i in range(0, jnp.size(knots)):
        labels.append('k'+str(i+1))
    mc_model.show_corner_plot(labels, truths=np.median(mc_model.sampler.flatchain, axis=0), quiet=True)
    plt.savefig("mcmc_results/"+name+"_cornerplot.png")

    # Plotting Chains
    n_cols = int((len(soln) + 2) / 3)
    fig, axes = plt.subplots(n_cols,3, figsize=(20,20))
    fig.subplots_adjust(hspace=0.5)
    for i in range(0, n_cols):
        for j in range(0, 3):
            if(3*i+j < len(soln)):
                axes[i][j].plot(np.linspace(0, nwalkers, niter), mc_model.sampler.get_chain()[:, :, 3*i+j].T)
                axes[i][j].set_ylim(BOUNDS[0][3*i+j], BOUNDS[1][3*i+j])
                axes[i][j].set_title(labels[3*i+j])
    plt.savefig("mcmc_results/"+name+"_chainplot.png")

    # Plotting Images
    sc_image = quick_image_cent(soln, PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = row["Distance"], knots=knots)
    mc_soln = np.median(mc_model.sampler.flatchain, axis=0)
    mc_image = quick_image_cent(mc_soln, PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = row["Distance"], knots=knots)
    plot_mc_img(target_image, sc_image, mc_image, round(-llp(soln), 3), round(-llp(mc_soln), 3))
    plt.savefig("mcmc_results/"+name+"_mcresid.png")

    return mc_soln, get_aic(-llp(mc_soln), len(mc_soln)), get_bic(-llp(mc_soln), len(mc_soln))


image_data = pd.read_csv('FitsResults/fits.csv')
image_data.columns = ["Name", "xc", "yc", "Alpha_In", "Alpha_Out", "Radius", "Inclination", "Position Angle", "Distance", "Knots"]
image_data.set_index("Name", inplace=True)
image_data.columns = ["xc", "yc", "Alpha_In", "Alpha_Out", "Radius", "Inclination", "Position Angle", "Distance", "Knots"]
print(image_data)
print()

# Gets rid of warnings
warnings.filterwarnings("ignore")

num_disks = len(image_data)

start = 7
#num_disks = start+1

for i in tqdm(range(start, num_disks)):
    name = image_data.index[i]
    row = image_data.loc[name]
    start = datetime.now()
    hdul = fits.open("Fits/"+name+".fits")
    target_image = process_image(hdul['SCI'].data[1,:,:])
    err_map = process_image(create_empirical_err_map(hdul['SCI'].data[2,:,:])).astype(jnp.float64)
    soln = fit_spline(row, target_image, err_map)
    mc_soln, aic, bic = run_mcmc_ab(soln, target_image, err_map, row, name)

    # Print Messages
    print(str(i+1) + " of " + str(num_disks) + " done.")
    print('Name: ' + str(name))
    print('Soln: ' + str(mc_soln))
    print('AIC: ' + str(aic))
    print('BIC: ' + str(bic))
    print('Time taken: ' + str(datetime.now()-start))
    print()


'''name = image_data.index[1]
row = image_data.loc[name]
start = datetime.now()
hdul = fits.open("Fits/"+name+".fits")
target_image = process_image(hdul['SCI'].data[1,:,:])
err_map = process_image(create_empirical_err_map(hdul['SCI'].data[2,:,:])).astype(jnp.float64)
soln = fit_spline(row, target_image, err_map)
mc_soln, aic, bic = run_mcmc_ab(soln, target_image, err_map, row, name)

ind += 1

# Print Messages
print(str(ind) + " of " + str(num_disks) + " done.")
print('Name: ' + str(name))
print('Soln: ' + str(mc_soln))
print('AIC: ' + str(aic))
print('BIC: ' + str(bic))
print('Time taken: ' + str(datetime.now()-start))
print()'''