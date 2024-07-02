import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

class MCMC_model():
    def __init__(self, fun, theta_bounds):
        self.fun = fun
        self.theta_bounds = theta_bounds
        self.sampler = None
        self.pos = None
        self.prob = None
        self.state = None

    def _lnprior(self, theta):
        if np.all(theta > self.theta_bounds[0]) and np.all(theta < self.theta_bounds[1]):
            return 0
        else:
            return -np.inf

    def _lnprob(self, theta):
        lp = self._lnprior(theta)
        if not lp == np.inf:
            return -np.inf
        return lp + self.fun(theta)

    def run(self, initial, nwalkers=500, niter = 500, nconst = 1e-7):
        ndim = len(initial)
        p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100)
        sampler.reset()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        self.sampler, self.pos, self.prob, self.state = sampler, pos, prob, state
        return sampler, pos, prob, state

    def get_theta_max(self):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        return self.sampler.flatchain[np.argmax(self.sampler.flatlnprobability)]

    def show_corner_plot(self, labels, truths=None, show_titles=True, plot_datapoints=True, quantiles = [0.16, 0.5, 0.84]):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        fig = corner.corner(self.sampler.flatchain,truths=truths, show_titles=show_titles,labels=labels,plot_datapoints=plot_datapoints,quantiles=quantiles)

    def plot_results(self, model):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        plt.ion()
        samples = self.sampler.flatchain
        print(len(samples[np.random.randint(len(samples), size=100)]))
        for theta in samples[np.random.randint(len(samples), size=100)]:
            plt.show(model(theta))
        plt.show()