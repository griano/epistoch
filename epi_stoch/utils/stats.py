# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:14:55 2020

@author: Germán Riaño
"""

import numpy as np

import pytest
import math
import warnings
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
    mu2 = mu*mu
    ln_mu = np.log( mu2 / np.sqrt(mu2 + sigma2) )
    ln_sigma = np.sqrt( np.log(1 + sigma2 / mu2) )
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
    alpha = (mu / sigma)**2
    beta = mu / alpha
    return stats.gamma(a= alpha, scale=beta)

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
        name = dist.dist.name
        if name == 'expon':
            return_fun = lambda x: np.exp(-x)
        if name == 'gamma':
            a = dist.kwds['a']
            return_fun = lambda x: a * stats.gamma.sf(x, a=a+1) - x * stats.gamma.sf(x, a)
        # Standard normal loss function used below
        if name == "norm":
            # loc and scale not set for the normal
            loc = dist.args[0] if len(dist.args) > 1 else None
            scale = dist.args[1] if len(dist.args) > 2 else None
            return_fun = lambda z: stats.norm.pdf(z) - z*stats.norm.sf(z)
        if name == "lognorm":
            def loss1(x):
                s = dist.kwds['s']
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    result = np.exp(.5 *s*s) * stats.norm.sf(np.log(x)/s - s) - x*stats.norm.sf(np.log(x)/s)
                return result
            return_fun = loss1
        loc = dist.kwds.get('loc', 0.0) if loc is None else loc
        scale = dist.kwds.get('scale', 1.0) if scale is None else scale
        return lambda x: np.where(x>lo, scale*return_fun((x - loc)/scale), dist.mean() - x)   
        
    return np.vectorize(lambda x : integrate.quad(dist.sf, x, hi)[0])


def avg_recovery_rate(dist):
    loss = loss_function(dist)
    def integrand(t):
        sf = dist.sf(t)
        lss = loss(t)
        result = np.zeros_like(t)
        valid = (lss > 0) 
        result[valid] = sf[valid] * sf[valid] / lss[valid]
        return  result
    gam = 1./dist.mean()
    a, b  = dist.support()
    result = gam * integrate.quad(integrand, a, b)[0]
    return result

def test_loss_function():
    EPS = 1e-5
    for force in [True, False]:
        expo = stats.expon(scale = 4)
        gm1 = stats.gamma(a=1, scale=4) # identical to previous one
        gm2 = stats.gamma(a=2, scale=3)
        nrm = stats.norm(loc=4, scale=1)
        lnrm = get_lognorm(4,1)
        loss_expo = loss_function(expo, force=force)
        loss_gm1 = loss_function(gm1, force=force)
        loss_gm2 = loss_function(gm2, force=force)
        loss_norm = loss_function(nrm, force=force)
        loss_lnrm = loss_function(lnrm, force=force)
        x = np.array([-2, -1, 0, 1, 3, 1000])
        # For positive vars loss1(0) = mean
        assert(4 == pytest.approx(loss_expo(0)))
        assert(4 == pytest.approx(loss_gm1(0)))
        assert(4 == pytest.approx(loss_lnrm(0)))
        loss_expo_at_x = np.array([6.00000000e+000, 5.00000000e+000, 4.00000000e+000, 3.11520313e+000, 1.88946621e+000, 1.06882573e-108])
        assert(loss_expo_at_x == pytest.approx(loss_expo(x)))
        assert(loss_expo_at_x == pytest.approx(loss_gm1(x)))
        loss_gamma2_at_x = np.array([8.00000000e+000, 7.00000000e+000, 6.00000000e+000, 5.01571917e+000,3.31091497e+000, 1.72896654e-142])
        assert(loss_gamma2_at_x == pytest.approx(loss_gm2(x)))
        loss_norm_at_x = np.array([6., 5, 4, 3.00038215, 1.08331547,0.])
        assert(loss_norm_at_x == pytest.approx(loss_norm(x), rel=EPS))
        loss_lnrm_at_x = np.array([6, 5, 4, 3.00000000e+000,1.05078013e+000, 7.72290351e-112])
        assert(loss_lnrm_at_x == pytest.approx(loss_lnrm(x), rel=EPS))
   
if __name__ == '__main__':
    test_loss_function()
    