# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:14:55 2020

@author: Germán Riaño
"""

import math
import warnings

import numpy as np
from scipy import integrate, stats


def get_lognorm(mu, sigma):
    """
    Builds a lognormal distribution from mean and standard deviation for this 
    variable. (Not the mean and sd of the corresponding normal)

    Parameters
    ----------
    mu : double
        The expected value
    sigma : double
        standard deviation

    Returns
    -------
    A frozen ditribution object

    """
    sigma2 = sigma * sigma
    mu2 = mu * mu
    ln_mu = np.log(mu2 / np.sqrt(mu2 + sigma2))
    ln_sigma = np.sqrt(np.log(1 + sigma2 / mu2))
    return stats.lognorm(s=ln_sigma, scale=math.exp(ln_mu))


def get_gamma(mu, sigma):
    """
    Builds a gamma distribution from mean and standard deviation for this 
    variable.

    Parameters
    ----------
    mu : double
        The expected value
    sigma : double
        standard deviation

    Returns
    -------
    A frozen ditribution object

    """
    alpha = (mu / sigma) ** 2
    beta = mu / alpha
    return stats.gamma(a=alpha, scale=beta)


class ConstantDist(stats.rv_continuous):
    def _cdf(self, x):
        return np.where(x >= 0, 1.0, 0.0)

    def _ppf(self, p):
        return 0.0

    def _loss1(self, x):
        return np.where(x > 0, 0.0, -x)  # [E(0-x)^+]


constant = ConstantDist(a=0.0, name="constant")


def loss_function(dist, force=False):

    """
    Creates a loss function of order 1 for a distribution from scipy

    Parameters
    ----------
    dist : scipy.stats._distn_infrastructure.rv_froze
        a distribution object form scipy.stats
    force : boolean
        whether force an integral computation instead of known formula        

    Returns
    -------
    Callable that represent this loss function

    """
    lo, hi = dist.support()
    loc, scale = None, None
    if not force:
        if "loss1" in dir(dist):
            return dist.loss1
        name = None
        return_fun = None
        if "name" in dir(dist):
            name = dist.name
        elif "dist" in dir(dist):
            if "name" in dir(dist.dist):
                name = dist.dist.name
        if name == "expon":
            return_fun = lambda x: np.exp(-x)
        if name == "gamma":
            a = dist.kwds["a"]
            return_fun = lambda x: a * stats.gamma.sf(x, a=a + 1) - x * stats.gamma.sf(x, a)
        # Standard normal loss function used below
        if name == "norm":
            # loc and scale not set for the normal
            loc = dist.args[0] if len(dist.args) > 1 else None
            scale = dist.args[1] if len(dist.args) > 2 else None
            return_fun = lambda z: stats.norm.pdf(z) - z * stats.norm.sf(z)
        if name == "lognorm":

            def loss1(x):
                s = dist.kwds["s"]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    result = np.exp(0.5 * s * s) * stats.norm.sf(np.log(x) / s - s) - x * stats.norm.sf(np.log(x) / s)
                return result

            return_fun = loss1
        if name == "constant":
            loc = dist.args[0] if len(dist.args) > 0 else None
            return_fun = dist.dist._loss1
        if return_fun is not None:
            loc = dist.kwds.get("loc", 0.0) if loc is None else loc
            scale = dist.kwds.get("scale", 1.0) if scale is None else scale
            loss1 = lambda x: np.where(x > lo, scale * return_fun((x - loc) / scale), dist.mean() - x)
            return loss1

    loss1 = np.vectorize(lambda x: integrate.quad(dist.sf, x, hi)[0])
    return loss1


def avg_recovery_rate(dist):
    loss = loss_function(dist)

    def integrand(t):
        sf = dist.sf(t)
        lss = loss(t)
        result = np.zeros_like(t)
        valid = lss > 0
        result[valid] = sf[valid] * sf[valid] / lss[valid]
        return result

    gam = 1.0 / dist.mean()
    a, b = dist.support()
    result = gam * integrate.quad(integrand, a, b)[0]
    return result
