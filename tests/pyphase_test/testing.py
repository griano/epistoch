# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:59:01 2020

@author: Germán Riaño, PhD
"""

import numpy as np
from scipy.stats import rv_continuous


class mixrv_gen(rv_continuous):
    def __init__(self, ps, vs, *args, **kwargs):
        super(mixrv_gen, self).__init__(*args, **kwargs)
        n = max(ps.shape)
        ps.shape = (n, 1)
        self.ps = ps
        self.vs = vs

    def _cdf(self, x):
        ps = self.ps
        vs = self.vs
        return np.sum(ps * np.array([v.cdf(x) for v in vs]), axis=0)

    def _munp(self, n):
        ps = np.transpose(self.ps)
        vs = self.vs
        return np.sum(ps * np.array([v.moment(n) for v in vs]))


mix_rv = mixrv_gen


def assert_dist_identical(dist1, dist2, name=None):
    print("Comparing Distributions " + (name if name is not None else ""))
    print("  Testing stats")
    np.testing.assert_allclose(dist1.stats(moments="mvsk"), dist2.stats(moments="mvsk"))
    print("  Testing CDF")
    p1 = np.linspace(0.0, 0.99)
    x1 = dist1.ppf(p1)
    p2 = dist2.cdf(x1)
    np.testing.assert_allclose(p1, p2)
    print("  Testing PDF")
    np.testing.assert_allclose(dist1.pdf(x1), dist2.pdf(x1))
