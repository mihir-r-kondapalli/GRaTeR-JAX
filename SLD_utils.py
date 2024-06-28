import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from interpolated_univariate_spline import InterpolatedUnivariateSpline
from interpolated_map_spline import interpolate as interpolate_map

class Jax_class:

    param_names = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array (params) and unpacks it into a
        dictionary with the parameter names as keys.
        """
        p_dict = {}
        keys = list(cls.param_names.keys())
        i = 0
        for i in range(0, len(p_arr)):
            p_dict[keys[i]] = p_arr[i]

        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.
        """    
        p_arrs = []
        for name in cls.param_names.keys():
            p_arrs.append(p_dict[name])
        return jnp.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """
    """

    param_names = {'ain': 5., 'aout': -5., 'a': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'amin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "apeak": 0., "apeak_surface_density": 0., "itiltthreshold": 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, accuracy=5.e-3, ain=5., aout=-5., a=60., e=0., ksi0=1., gamma=2., beta=1., amin=0., dens_at_r0=1.):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane
        """

        p_dict = {}
        p_dict["accuracy"] = accuracy

        p_dict["ksi0"] = ksi0
        p_dict["gamma"] = gamma
        p_dict["beta"] = beta
        p_dict["zmax"] = ksi0*(-jnp.log(p_dict["accuracy"]))**(1./gamma)

        # Set Vertical Density Analogue
        gamma = jnp.where(gamma < 0., 0.1, gamma)
        ksi0 = jnp.where(ksi0 < 0., 0.1, ksi0)
        beta = jnp.where(beta < 0., 0., beta)

        # Set Radial Density Analogue
        ain = jnp.where(ain < 0.1, 0.1, ain)
        aout = jnp.where(aout > -0.1, -0.1, aout)
        e = jnp.where(e < 0., 0., e)
        e = jnp.where(e >= 1, 0.99, e)
        amin = jnp.where(amin < 0., 0., amin)
        dens_at_r0 = jnp.where(dens_at_r0 < 0., 1., dens_at_r0)

        p_dict["ain"] = ain
        p_dict["aout"] = aout
        p_dict["a"] = a
        p_dict["e"] = e
        p_dict["p"] = p_dict["a"]*(1-p_dict["e"]**2)
        p_dict["amin"] = amin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["amin"]*(1-p_dict["e"]**2)
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["a"]*p_dict["accuracy"]**(1/p_dict["aout"])
        p_dict["apeak"] = p_dict["a"] * jnp.power(-p_dict["ain"]/p_dict["aout"],
                                        1./(2.*(p_dict["ain"]-p_dict["aout"])))
        Gamma_in = p_dict["ain"]+p_dict["beta"]
        Gamma_out = p_dict["aout"]+p_dict["beta"]
        p_dict["apeak_surface_density"] = p_dict["a"] * jnp.power(-Gamma_in/Gamma_out,
                                                        1./(2.*(Gamma_in-Gamma_out)))
        # the above formula comes from Augereau et al. 1999.
        p_dict["itiltthreshold"] = jnp.rad2deg(jnp.arctan(p_dict["rmax"]/p_dict["zmax"]))

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        distr = cls.unpack_pars(distr_params)

        radial_ratio = r*(1-distr["e"]*costheta)/((distr["p"])+1e-8)

        den = (jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["ain"]) +
               jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["aout"]))
        radial_density_term = jnp.sqrt(2./den)*distr["dens_at_r0"]
        #if distr["pmin"] > 0:
        #    radial_density_term[r/(distr["pmin"]/(1-distr["e"]*costheta)) <= 1] = 0
        radial_density_term = jnp.where(distr["pmin"] > 0, 
                                        jnp.where(r*(1-distr["e"]*costheta)/((distr["p"])+1e-8) <= 1, 0., radial_density_term),
                                        radial_density_term)

        den2 = (distr["ksi0"]*jnp.power(radial_ratio, distr["beta"]))
        vertical_density_term = jnp.exp(-jnp.power(jnp.abs(z)/(den2+1e-8), distr["gamma"]))
        return radial_density_term*vertical_density_term

class HenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a single Henyey
    Greenstein function.
    """

    param_names = {'g': 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        Constructor of a Heyney Greenstein phase function.

        Parameters
        ----------
        spf_dico :  dictionnary containing the key "g" (float)
            g is the Heyney Greenstein coefficient and should be between -1
            (backward scattering) and 1 (forward scattering).
        """

        p_dict = {}
        g = func_params[0]
        g = jnp.where(g>=1, 0.99, g)
        g = jnp.where(g<=-1, -0.99, g)
        p_dict["g"] = g

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        p_dict = cls.unpack_pars(phase_func_params)
        
        return 1./(4*jnp.pi)*(1-p_dict["g"]**2) / \
            (1+p_dict["g"]**2-2*p_dict["g"]*cos_phi)**(3./2.)


class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.
    """

    param_names = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        """

        p_dict = {}
        p_dict['g1'] = func_params[0]
        p_dict['g2'] = func_params[1]
        p_dict['weight'] = func_params[2]

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """

        p_dict = cls.unpack_pars(phase_func_params)

        hg1 = p_dict['weight'] * 1./(4*jnp.pi)*(1-p_dict["g1"]**2) / \
            (1+p_dict["g1"]**2-2*p_dict["g1"]*cos_phi)**(3./2.)
        hg2 = (1-p_dict['weight']) * 1./(4*jnp.pi)*(1-p_dict["g2"]**2) / \
            (1+p_dict["g2"]**2-2*p_dict["g2"]*cos_phi)**(3./2.)
        
        return hg1+hg2
    

# Uses 10 knots by default
# Values must be cos(phi) not phi
class InterpolatedUnivariateSpline_SPF(Jax_class):
    """
    Implementation of a spline scattering phase function. Uses 10 knots by default, takes 10 y values as parameters.
    Locations are fixed to linspace(0, pi, knots), pack_pars and init both return the spline model itself
    """

    param_names = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        return p_arr

    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3))
    def pack_pars(cls, p_arr, inc = 0, knots=10):
        """
        This function takes a array of (knots) values and converts them into an InterpolatedUnivariateSpline model.
        Also has inclination bounds which help narrow the spline fit
        """    
        
        x_vals = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), knots)
        y_vals = p_arr
        return InterpolatedUnivariateSpline(x_vals, y_vals)

    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3))
    def init(cls, p_arr, inc = 0, knots=10):
        """
        """

        x_vals = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), knots)
        y_vals = p_arr
        return InterpolatedUnivariateSpline(x_vals, y_vals)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, spline_model, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        spline_model : InterpolatedUnivariateSpline
            spline model to represent scattering light phase function
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        
        return spline_model(cos_phi)
    

# Only works if input y values were from x values from 1 to -1
# Uses 10 knots by default
# Values must be cos(phi) not phi
class InterpolatedMapSpline_SPF(Jax_class):
    """
    Implementation of a spline scattering phase function. Uses 10 knots by default, takes 10 y values as parameters.
    Locations are fixed to linspace(0, pi, knots), pack_pars and init both return the spline model itself
    """

    param_names = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        return p_arr

    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3,4))
    def pack_pars(cls, p_arr, inc = 0, knots=10, precision = 1000):
        """
        This function takes a array of (knots) values and converts them into an InterpolatedUnivariateSpline model.
        Also has inclination bounds which help narrow the spline fit
        """    
        
        x_vals = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), knots)
        y_vals = p_arr
        x_coords = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), precision)
        return interpolate_map(x_vals, y_vals, x_coords)

        # Return y_values of spline fit along with inc (if inc was not 0)
        #return jnp.concatenate([precision, inc, interpolate_map(x_vals, y_vals, x_coords)])

    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3,4))
    def init(cls, p_arr, inc = 0, knots=10, precision = 1000):
        """
        """

        x_vals = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), knots)
        y_vals = p_arr
        x_coords = jnp.linspace(jnp.cos(inc), jnp.cos(jnp.pi-inc), precision)
        return interpolate_map(x_vals, y_vals, x_coords)

        # Return y_values of spline fit along with inc (if inc is not 0)
        #return jnp.concatenate([precision, inc, interpolate_map(x_vals, y_vals, x_coords)])
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, arr, cos_phi, precision=1000):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        arr : jax array 2d
            array of interpolated xs and ys
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """

        return jnp.take(arr, precision-1-((1+cos_phi)/2*precision).astype(jnp.int32))

        # If inc is not 0
        # return jnp.take(arr, [jnp.round((cos_phi-arr[0])*precision)+1])