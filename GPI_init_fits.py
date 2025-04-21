import os
os.environ["WEBBPSF_PATH"] = 'webbpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'webbpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]='False'

from utils.objective_functions import jax_model, objective_model, objective_fit, log_likelihood, Parameter_Index
from utils.new_SLD_utils import DoubleHenyeyGreenstein_SPF, InterpolatedUnivariateSpline_SPF, EMP_PSF, DustEllipticalDistribution2PowerLaws, Winnie_PSF
from utils.SLD_ojax import ScatteredLightDisk
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from astropy.io import fits
import pandas as pd
from statistical_analysis.optimize_framework import OptimizeUtils
from scipy.optimize import minimize
jax.config.update("jax_debug_nans", True)
import pickle

def read_csv_gpi(file):
    """reads in csv of initial guesses"""
    image_data = pd.read_csv(file)
    image_data.set_index("Name", inplace=True)
    image_data.columns = ["Radius", "Inclination", "Position Angle", "Distance","Knots"]    
    print('csv file of initial guesses read')
    return image_data

def fit_disk(name):
    """fits a gpi disk and returns best params"""

    print('working on initial minimization fit for ',name)

    image_data = read_csv_gpi('GPI_data/image_info_filt.csv')
    row = image_data.loc[name]

    fits_image_filepath = "GPI_data/" + name + ".fits"
    hdul = fits.open(fits_image_filepath)

    target_image = OptimizeUtils.process_image(hdul['SCI'].data[1,:,:])
    err_map = OptimizeUtils.process_image(OptimizeUtils.create_empirical_err_map(hdul['SCI'].data[2,:,:])) 

    spf_params = InterpolatedUnivariateSpline_SPF.params
    psf_params = EMP_PSF.params
    spf_params['knot_values'] = jnp.ones(6)
    spf_params['num_knots'] = 6

    disk_params = Parameter_Index.disk_params
    disk_params['sma'] = row["Radius"]
    disk_params['inclination'] = row["Inclination"]
    disk_params['position_angle'] = row["Position Angle"]

    misc_params = Parameter_Index.misc_params
    misc_params['distance'] = row["Distance"]

    fit_keys = ['sma', 'inclination', 'position_angle', 'alpha_in','alpha_out','ksi0','gamma','beta','x_center','y_center','knot_values']
    print('fitting for',fit_keys)

    llp = lambda x: -objective_fit([x[0], x[1], x[2], x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10:]], fit_keys, disk_params, spf_params, psf_params, misc_params,
                                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, EMP_PSF, target_image,
                                err_map)

    start_disk_params = disk_params
    start_disk_params['sma'] = disk_params['sma']
    start_disk_params['inclination'] = disk_params['inclination']
    start_disk_params['position_angle'] = disk_params['position_angle']

    init_x = jnp.concatenate([jnp.array([start_disk_params['sma'], start_disk_params['inclination'], start_disk_params['position_angle'],5,-5,3,2,1,70,70]), 0.5*jnp.ones(11)])

    soln = minimize(llp, init_x, options={'disp': True, 'max_itr': 500})

    print('minimization complete. solution:')
    print(soln.x)

    with open('GPI_results/'+name+'_init.pickle', 'wb') as file:
        pickle.dump(soln.x, file)

    print("values written to ",'GPI_results/'+name+"_init.pickle")

    ##add figure saving
    