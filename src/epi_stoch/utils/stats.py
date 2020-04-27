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
from pyphase.ph import ph_erlang


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

class ConstantDist(stats.rv_continuous):
    
    def _cdf(self, x):
        return np.where(x>=0, 1., 0.)

    def _ppf(self, p):
        return 0.0
    
    def _loss1(self, x):
        return np.where(x>0, 0.0, -x)  # [E(0-x)^+] 

constant = ConstantDist(a =0., name="constant")

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
        if 'loss1' in dir(dist):
            return dist.loss1
        name = None
        return_fun = None
        if 'name'in dir(dist):
            name = dist.name
        elif 'dist' in dir(dist):
            if 'name' in dir(dist.dist):
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
        if name == "constant":
            loc = dist.args[0] if len(dist.args) > 0 else None
            return_fun = dist.dist._loss1
        if return_fun is not None:
            loc = dist.kwds.get('loc', 0.0) if loc is None else loc
            scale = dist.kwds.get('scale', 1.0) if scale is None else scale
            loss1 = lambda x: np.where(x>lo, scale*return_fun((x - loc)/scale), dist.mean() - x) 
            return loss1
        
    loss1 = np.vectorize(lambda x : integrate.quad(dist.sf, x, hi)[0])
    return loss1

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


###############################
# TEST: To be moved later
###############################

def test_constant():
    v = constant()
    assert 0 == v.mean()
    v = constant(loc=8)
    assert 8 == v.mean()
    const_loss = loss_function(v)
    # assert 1 == const_loss(9)
    

def test_loss_function():
    EPS = 1e-5
    expo = stats.expon(scale = 4)
    gm1 = stats.gamma(a=1, scale=4) # identical to previous one
    gm2 = stats.gamma(a=2, scale=3)
    nrm = stats.norm(loc=4, scale=1)
    lnrm = get_lognorm(4,1)
    const = constant(loc=4)
    for force in [True, False]:
        loss_expo = loss_function(expo, force=force)
        loss_gm1 = loss_function(gm1, force=force)
        loss_gm2 = loss_function(gm2, force=force)
        loss_norm = loss_function(nrm, force=force)
        loss_lnrm = loss_function(lnrm, force=force)
        loss_const = loss_function(const, force=force)
        assert(4 == pytest.approx(loss_expo(0)))
        assert(4 == pytest.approx(loss_gm1(0)))
        assert(4 == pytest.approx(loss_lnrm(0)))
        assert(4 == pytest.approx(loss_const(0)))
        x = np.array([-2, -1, 0, 1, 3, 1000])
        # For positive vars loss1(0) = mean
        loss_expo_at_x = np.array([6.00000000e+000, 5.00000000e+000, 4.00000000e+000, 3.11520313e+000, 1.88946621e+000, 1.06882573e-108])
        np.testing.assert_allclose(loss_expo(x),loss_expo_at_x,  atol=EPS, err_msg=f"Expo loss, force={force}")
        np.testing.assert_allclose(loss_gm1(x), loss_expo_at_x, atol=EPS, err_msg=f"Gamma1 loss, force={force}")
        loss_gamma2_at_x = np.array([8.00000000e+000, 7.00000000e+000, 6.00000000e+000, 5.01571917e+000,3.31091497e+000, 1.72896654e-142])
        np.testing.assert_allclose(loss_gm2(x), loss_gamma2_at_x, atol=EPS)
        loss_norm_at_x = np.array([6., 5, 4, 3.00038215, 1.08331547,0.])
        assert(loss_norm_at_x == pytest.approx(loss_norm(x), rel=EPS))
        loss_lnrm_at_x = np.array([6, 5, 4, 3.00000000e+000,1.05078013e+000, 7.72290351e-112])
        assert(loss_lnrm_at_x == pytest.approx(loss_lnrm(x), rel=EPS))
        loss_const_at_x = np.array([6., 5., 4., 3., 1., 0.])
        np.testing.assert_allclose(loss_const(x), loss_const_at_x, err_msg=f"Constant loss, force={force}" )

def test_ph_loss():
    phd = ph_erlang(3, 10)
    gma = stats.gamma(a=3, scale=1/10)
    x = np.linspace(0,2)
    loss_gma = loss_function(gma, force=False)
    loss_gma_at_x = loss_gma(x)
    loss_ph_at_x = phd.loss1(x)
    np.testing.assert_allclose(loss_gma_at_x, loss_ph_at_x)
    

if __name__ == '__main__':
    test_constant()
    test_loss_function()
    test_ph_loss()