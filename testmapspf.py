from interpolated_map_spline import interpolate
from SLD_utils import DoubleHenyeyGreenstein_SPF, InterpolatedMapSpline_SPF
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

KNOTS = 10

# Initial spline guess
init_spf_params = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}
init_spf = DoubleHenyeyGreenstein_SPF.init([init_spf_params['g1'], init_spf_params['g2'], init_spf_params['weight']])
init_x = np.linspace(np.cos(0), np.cos(np.pi), KNOTS)
init_y = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(init_spf, init_x)
init_xs = np.linspace(np.cos(0), np.cos(np.pi), 100)
init_ys = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(init_spf, init_xs)

disk_params2 = {}
disk_params2['inclination'] = 40. #In degrees
disk_params2['position_angle'] = 30. #In Degrees
disk_params2['alpha_in'] = 5. #The inner power law
disk_params2['alpha_out'] = -7. #The outer power law
#gs_ws = jnp.array([0.8,-0.2,0,0.75,0.25,0.]) #Here we have 3 Henyey-Greenstein functions with g parameters of 1, -1, and 0. The weights are 0.75, 0.25, and 0 respectively. 
disk_params2['flux_scaling'] = 1e6

#The disk size
disk_params2['sma'] = 40. #This is the semi-major axis of the model in astronomical units. 
#To get this in pixels, divide by the distance to the star, to get it in arcseconds. To get it in pixeks, divide by the pixel scale.

# DHG parameters
spf_params = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

inc = disk_params2['inclination']

# Initial spline guess
init_spf_params = {'g1': 0.5, 'g2': -0.5, 'weight': 0.5}
init_spf = DoubleHenyeyGreenstein_SPF.init([init_spf_params['g1'], init_spf_params['g2'], init_spf_params['weight']])
KNOTS = 10
knot_xs = jnp.array(np.linspace(np.cos(0), np.cos(np.pi), KNOTS))
knot_ys = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(init_spf, knot_xs)
full_xs = jnp.array(np.linspace(np.cos(0), np.cos(np.pi), 100))
full_ys = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(init_spf, full_xs)

ims = InterpolatedMapSpline_SPF.init(knot_ys)
fin_ys = InterpolatedMapSpline_SPF.compute_phase_function_from_cosphi(ims, full_xs)

plt.scatter(knot_xs, knot_ys, color = 'g')
plt.plot(full_xs, full_ys, color = 'g')
plt.plot(full_xs, fin_ys, color = 'b')

plt.xlabel("cos(phi)")
plt.ylabel("Scattered Light Value")
plt.show()
plt.savefig('spftest.png')