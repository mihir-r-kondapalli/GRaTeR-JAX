import os
os.environ["WEBBPSF_PATH"] = 'webbpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'webbpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"
from astropy.io import fits
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
jax.config.update("jax_enable_x64", True)
import pandas as pd
from statistical_analysis.optimize_framework import Optimizer, OptimizeUtils
from utils.objective_functions import objective_model, objective_ll, objective_fit, Parameter_Index
from utils.SLD_ojax import ScatteredLightDisk
from utils.SLD_utils import *
    
def init_fit(name,spf_type='spline',plot=True,save=False,num_knots=6):
    fits_image_filepath = "/home/blewis/GPI_data/" + str(name) + ".fits"
    hdul = fits.open(fits_image_filepath)
    target_image = OptimizeUtils.process_image(hdul['SCI'].data[1,:,:],bounds=(50, 230, 50, 230))
    err_map = OptimizeUtils.process_image(OptimizeUtils.create_empirical_err_map(hdul['SCI'].data[2,:,:]),bounds=(50, 230, 50, 230)) #, outlier_pixels=[(57, 68)]))
    if spf_type=='spline':
        spf_params = InterpolatedUnivariateSpline_SPF.params
        psf_params = EMP_PSF.params
        disk_params = Parameter_Index.disk_params
        misc_params = Parameter_Index.misc_params
        misc_params['nx'] = 180
        misc_params['ny'] = 180

        image_data = pd.read_csv('statistical_analysis/image_info_filt.csv')
        image_data.set_index("Name", inplace=True)
        image_data.columns = ["Radius", "Inclination", "Position Angle", "Distance", "Knots", "a_in", "a_out", "eccentricity", "ksi0", "gamma", "beta", "omega", "x_center", "y_center"]
        row = image_data.loc[name]

        disk_params['sma'] = row["Radius"]
        disk_params['inclination'] = row["Inclination"]
        disk_params['position_angle'] = row["Position Angle"]
        misc_params['distance'] = row["Distance"]
        spf_params['num_knots'] = num_knots
        spf_params['knot_values'] = jnp.full(spf_params['num_knots'],0.5)
        disk_params['alpha_in'] = row['a_in']
        disk_params['alpha_out'] = row['a_out']
        disk_params['e'] = row['eccentricity']
        disk_params['ksi0'] = row['ksi0']
        disk_params['gamma'] = row['gamma']
        disk_params['beta'] = row['beta']
        disk_params['omega'] = row['omega']
        disk_params['x_center'] = row['x_center']
        disk_params['y_center'] = row['y_center']

        opt = Optimizer(disk_params, spf_params, psf_params, misc_params, 
                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, EMP_PSF)
        fit_keys = ['alpha_in', 'alpha_out', 'sma', 'e', 'ksi0','gamma','beta','omega','inclination', 'position_angle', 'x_center', 'y_center', 'knot_values']
        opt.inc_bound_knots()
        opt.scale_knots(target_image)
        soln = opt.scipy_optimize(fit_keys, target_image, err_map, disp_soln=True, iters = 1000)
        optimal_image = opt.model()
        optimal_ll = opt.log_likelihood(target_image,err_map)
        titles = 'alpha_in, alpha_out, sma, e, ksi0, gamma, beta, omega, inclination, position_angle, x_center, y_center, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11'
        if save==True:
            np.savetxt('{}_init_fit_params.txt'.format(name),soln.x,header=titles)
    if plot==True:
        fig, axes = plt.subplots(1,3, figsize=(20,10))

        mask = OptimizeUtils.get_mask(target_image)
        vmin = np.nanpercentile(target_image[mask], 1)
        vmax = np.nanpercentile(target_image[mask], 99)

        xmax=70
        extent = [-(xmax*14.1)/2000, (xmax*14.1)/2000, (xmax*14.1)/2000, -(xmax*14.1)/2000]

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)

        im = axes[0].imshow(target_image, origin='lower', cmap='inferno',extent=extent)
        axes[0].set_title("Data",fontsize=16)
        axes[0].set_ylabel('$\Delta$RA (arcsec)',fontsize=14)
        axes[0].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
        #plt.colorbar(im, ax=axes[0], shrink=0.5)
        im.set_clim(vmin, vmax)

        im = axes[1].imshow(optimal_image, origin='lower', cmap='inferno',extent=extent)
        axes[1].set_title("Model",fontsize=16)
        #plt.colorbar(im, ax=axes[1], shrink=0.5)
        im.set_clim(vmin, vmax)
        axes[1].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)

        im = axes[2].imshow(target_image-optimal_image, origin='lower', cmap='inferno',extent=extent)
        axes[2].set_title("Residuals",fontsize=16)
        im.set_clim(vmin, vmax)
        axes[2].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
        plt.tight_layout()
        cb = plt.colorbar(im, ax=axes, shrink=0.5,pad=0.01)
        cb.set_label('Arbitrary Flux Units',fontsize=14)#,rotation=270)
        cb.ax.tick_params(labelsize=12)
        plt.savefig('{}_initial_fit.png'.format(name))

    return optimal_image, soln.x, optimal_ll