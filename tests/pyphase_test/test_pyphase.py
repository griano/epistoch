import numpy as np
from numpy import matlib as ml
from scipy.stats import expon, gamma

from pyphase.phase import ph_erlang, ph_expon, ph_mix, ph_sum, phase
from tests.pyphase_test.testing import assert_dist_identical, mix_rv


def test_ph_expo():
    lambd = 10
    exp = expon(scale=1.0 / lambd)
    dists = dict()
    dists["np"] = phase(alpha=np.array([1.0]), A=np.array([-lambd]))
    dists["float"] = phase(alpha=1.0, A=-lambd)
    dists["mat"] = phase(alpha=ml.mat([1.0]), A=ml.mat([[-lambd]]))
    dists["method"] = ph_expon(lambd)
    for name, dist in dists.items():
        assert_dist_identical(exp, dist, "Expo-" + name)


def test_ph_gamma():
    lam = 20.0
    n = 2
    gam = gamma(a=n, scale=1 / lam)
    a = [1.0, 0.0]
    A = [[-lam, lam], [0, -(lam)]]
    dists = dict()
    dists["list"] = phase(alpha=a, A=A)
    dists["np"] = phase(alpha=np.array(a), A=np.array(A))
    dists["mat"] = phase(alpha=ml.mat(a), A=ml.mat(A))
    dists["method"] = ph_erlang(2, lam)
    for name, dist in dists.items():
        assert_dist_identical(gam, dist, "Gamma-" + name)


def test_ph_sum():
    lam = 3.0
    v1 = phase(alpha=np.array([1.0]), A=np.array([-lam]))
    v2 = v1
    v = ph_sum(v1, v2)
    print(f"sum = {v}")
    expected = gamma(a=2, scale=1 / lam)
    assert_dist_identical(expected, v, "Sum Expos")


def test_ph_mix():
    lam1 = 4.0
    lam2 = 2.0
    p = 0.4
    v1 = ph_expon(lam1)
    v2 = ph_expon(lam2)
    v = ph_mix(v1, v2, p)
    print(f"mix = {v}")
    expected = mix_rv(ps=np.array([p, 1 - p]), vs=[v1, v2])
    x = np.array([1, 2, 3])
    print(f"v-cdf = {v.cdf(x)}")
    print(f"ex-cdf = {expected.cdf(x)}")
    print(f"v-moms: {[{n:v.moment(n) for n in range(1,4)}]}")
    print(f"exp-moms: {[{n:expected.moment(n) for n in range(1,4)}]}")
    assert_dist_identical(expected, v, "Mix-HyperExpo")


def test_loss1():
    lambd = 0.5
    v1 = ph_expon(lambd)
    x = np.linspace(1, 5, 1)
    exp_loss1 = lambda x: np.exp(-lambd * x) / lambd  # Noqa 731
    np.testing.assert_allclose(v1.loss1(x), exp_loss1(x))


def test_equilibrium_phases():
    v = ph_erlang(n=3, mean=3)
    print(v)
    pi = v.equilibrium_pi()
    print(pi)


#######################################
#  T E S T I N G
#######################################

if __name__ == "__main__":
    test_equilibrium_phases()
    test_ph_expo()
    test_ph_gamma()
    test_loss1()
    test_ph_sum()
    test_ph_mix()
