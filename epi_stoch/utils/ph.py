# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:55:02 2020

@author: Germán Riaño
"""

import numpy as np
from numpy import matlib as ml
from scipy.special import binom
from scipy.stats import rv_continuous, expon, gamma
from butools.ph import CheckPHRepresentation, CdfFromME, PdfFromME, MomentsFromME, CdfFromPH, PdfFromPH, MomentsFromPH
import butools

butools.verbose = True
butools.check = True

class PhaseTypeGen(rv_continuous):
    
    def __init__(self, alpha, A, prec=None):
        alpha = ml.mat(alpha)
        A = ml.mat(A)
        CheckPHRepresentation(alpha, A, prec)
        super(PhaseTypeGen, self).__init__(momtype=1, a=0, b=None, xtol=1e-14,
                 badvalue=None, name="phase", longname=None,
                 shapes=None, extradoc=None, seed=None)
        self.alpha = alpha
        self.A = A
        
    def _cdf(self, x):
        return  CdfFromPH(self.alpha, self.A, x)
        
    def _pdf(self, x):
        return PdfFromPH(self.alpha, self.A, x)
    
    def _stats(self):
        # Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
        alpha = self.alpha; A = self.A
        print(f"alpha = {alpha}, A = {A}.")
        moments = MomentsFromME(self.alpha, self.A, 4)
        moments = np.concatenate(( [1.0], moments))
        mean = moments[1]
        centered_moments ={n:sum([ binom(n,k)* moments[k] * (-mean)**(n-k)  for k in range(0,n+1)]) for n in range(1,5) } 
        var = centered_moments[2]
        sd = np.sqrt(var)
        skew = centered_moments[3] / sd**3
        # kurtosis is fourth central moment / variance**2 - 3
        kurt = centered_moments[4] / (var*var) - 3.
        return mean, var, skew, kurt

phase = PhaseTypeGen


def test_identical(dist1, dist2):
    print('  Testing stats')
    np.testing.assert_allclose(dist1.stats(moments='mvsk'), dist2.stats(moments='mvsk'))
    print('  Testing CDF')
    p1 = np.linspace(0.0, 0.99)
    x1 = dist1.ppf(p1)
    p2 = dist2.cdf(x1)
    np.testing.assert_allclose(p1, p2)
    print('  Testing PDF')
    np.testing.assert_allclose(dist1.pdf(x1), dist2.pdf(x1))
    
    

def test_ph_expo():
    lambd = 10
    exp = expon(scale=1./lambd)
    dists = dict()
    dists['np'] = phase(alpha=np.array([1.0]), A =np.array([-lambd]) )
    dists['float'] = phase(alpha=1.0, A =-lambd )
    dists['ml'] = phase(alpha=ml.mat([1.0]), A =ml.mat([[-lambd]]) )
    for name, dist in dists.items():
        print(f"Testing {name}")
        test_identical(exp, dist)
    
def test_ph_gamma():
    lam = 20.
    n = 2
    gam = gamma(a=n, scale = 1/lam)
    a = [1.0, 0.0]
    A = [[-lam, lam], [0, -(lam)]]
    dists = dict()
    dists['np'] = phase(alpha=np.array(a), A =np.array(A) )
    dists['ml'] = phase(alpha=ml.mat(a), A =ml.mat(A) )
    for name, dist in dists.items():
        print(f"Testing {name}")
        test_identical(gam, dist)

if __name__ == "__main__":
    test_ph_expo()
    test_ph_gamma()