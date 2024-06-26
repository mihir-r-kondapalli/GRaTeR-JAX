import jax
import jax.numpy as jnp
import numpy as np
from SLD_utils import DoubleHenyeyGreenstein_SPF
import matplotlib.pyplot as plt
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize
import time


# x is numpy array with 1 parameter, y is numpy array with 2 parameters
# y (peak, 1st offset peaks, 2nd offset peaks)
# Init Guess: x: [0, pi/2, pi], y: [0.3, 0.05, 0.1]
def get_spline_func(x, y):
    return InterpolatedUnivariateSpline(x, y)


KNOTS = 10

dspf = DoubleHenyeyGreenstein_SPF.init([0.5, -0.3, 0.7])

phis = np.linspace(0, np.pi, 100)
spf_vals = np.zeros(100)

for i in range(0, len(phis)):
    spf_vals[i] = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(dspf, np.cos(phis[i]))

rng = np.random.default_rng()
noise_level = 0.05
noise = np.random.normal(0, noise_level, KNOTS)
init_spf = DoubleHenyeyGreenstein_SPF.init([0.5, -0.3, 0.7])

x = np.linspace(0, np.pi, KNOTS)
y = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(init_spf, np.cos(x)) + noise
params = y

def get_error(y, x, phis, spf_vals):
    spline_func = get_spline_func(x, y)
    return np.sum(np.abs(np.power(spf_vals - spline_func(phis), 2)))

err_func = lambda vals: get_error(vals, x, phis, spf_vals)
start = time.time()
soln = minimize(err_func, params, method = None)
end = time.time()
spline_func = get_spline_func(x, soln.x)

plt.plot(phis, spf_vals)

plt.plot(phis, get_spline_func(x, y)(phis), color = 'g')
plt.scatter(x, y, color = 'g')
print('Initial Error: ' + str(get_error(y, x, phis, spf_vals)))

plt.plot(phis, spline_func(phis), color = 'r')
plt.scatter(x, soln.x, color = 'r')
print('Final Error: ' + str(get_error(soln.x, x, phis, spf_vals)))

plt.title("DHG Function vs Spline Model")
plt.show()
plt.savefig('test.png')

print('Time taken: ' + str(end-start))

print(spline_func._compute_coeffs(x))
