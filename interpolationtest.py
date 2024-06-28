from interpolated_map_spline import interpolate
from SLD_utils import DoubleHenyeyGreenstein_SPF
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


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

fin_knots = interpolate(knot_xs, knot_ys, knot_xs)
fin_ys = interpolate(knot_xs, knot_ys, full_xs)

plt.scatter(knot_xs, knot_ys, color = 'g')
plt.plot(full_xs, full_ys, color = 'g')
plt.scatter(knot_xs, fin_knots, color = 'b')
plt.plot(full_xs, fin_ys, color = 'b')

plt.xlabel("cos(phi)")
plt.ylabel("Scattered Light Value")
plt.title("Intial Spline vs Final Spline")
plt.show()
plt.savefig('test.png')