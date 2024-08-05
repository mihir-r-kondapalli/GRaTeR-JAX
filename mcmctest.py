
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

from mctest_utils import (process_image, create_empirical_err_map, get_inc_bounded_knots, get_aic, get_bic,
                          transpose_even_to_odd_spline, plot_mc_img)

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

jax.config.update("jax_enable_x64", True)

###############################################################################

def fit_spline(row, target_image, err_map, disp = False, flux_scaling=1e6):

    knots = get_inc_bounded_knots(row["Inclination"], row["Radius"], buffer = 0, num_knots=int(row["Knots"]))

    init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
    disk_pars = jnp.array([row["xc"], row["yc"], row["Alpha_In"], row["Alpha_Out"],
                        row["Radius"], row["Inclination"], row["Position Angle"]])

    bounds = []
    for i in range(0, int(row["Knots"])):
        bounds.append((1e-8, 0.1))

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

    llp = lambda x: -log_likelihood_1d_pos_cent(jnp.concatenate([x[0:7], jnp.exp(x[7:])]), 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        1e6, target_image, err_map, PSFModel = EMP_PSF, pxInArcsec=0.01414,
                        distance = row["Distance"], knots=knots)


    DISK_BOUNDS = np.array([np.array([0.1, -15, 0, 0, 0]), np.array([15, -0.1, 150, 180, 400])])
    CENT_BOUNDS = np.array([np.array([65, 65]), np.array([75, 75])])
    SPLINE_BOUNDS = np.array([1e-8 * np.ones(jnp.size(knots)), 0.1 * np.ones(jnp.size(knots))])
    BOUNDS = np.array([np.concatenate([CENT_BOUNDS[0], DISK_BOUNDS[0], np.log(SPLINE_BOUNDS[0])]),
                        np.concatenate([CENT_BOUNDS[1], DISK_BOUNDS[1], np.log(SPLINE_BOUNDS[1])])])

    if np.all(soln < BOUNDS[0]) and np.all(soln > BOUNDS[1]):
        print()
        print("Intial params out of bounds for MCMC!!!")
        print(soln)
        print(BOUNDS)
        oob = (soln < BOUNDS[0]) and (soln > BOUNDS[1])
        print(oob)
        exit()

    init_soln = jnp.concatenate([soln[0:7], jnp.log(soln[7:])])

    mc_model = MCMC_model(llp, BOUNDS)
    mc_model.run(init_soln, nconst = 1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns)

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

    # Need to adjust for log
    mc_image = quick_image_cent(jnp.concatenate([mc_soln[0:7], jnp.exp(mc_soln[7:])]), PSFModel = EMP_PSF, pxInArcsec=0.01414, distance = row["Distance"], knots=knots)

    plot_mc_img(name, err_map, target_image, sc_image, mc_image, round(-llp(init_soln), 3), round(-llp(mc_soln), 3))
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

start = 0
#num_disks = start+1

for i in tqdm(range(start, num_disks)):
    name = image_data.index[i]
    row = image_data.loc[name]
    start = datetime.now()
    hdul = fits.open("Fits/"+name+".fits")
    target_image = process_image(hdul['SCI'].data[1,:,:])
    err_map = process_image(create_empirical_err_map(hdul['SCI'].data[2,:,:])).astype(jnp.float64)
    soln = fit_spline(row, target_image, err_map)
    mc_soln, aic, bic = run_mcmc_ab(soln, target_image, err_map, row, name, nwalkers = 300, niter = 300, burns = 60)

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