import numpy as np
import pytest
from scipy import stats

from epistoch.utils.stats import constant, get_lognorm, loss_function
from pyphase.phase import ph_erlang


def test_constant():
    v = constant()
    assert 0 == v.mean()
    v = constant(loc=8)
    assert 8 == v.mean()
    const_loss = loss_function(v)
    assert 1 == const_loss(7)


def test_loss_function():
    EPS = 1e-5
    expo = stats.expon(scale=4)
    gm1 = stats.gamma(a=1, scale=4)  # identical to previous one
    gm2 = stats.gamma(a=2, scale=3)
    nrm = stats.norm(loc=4, scale=1)
    lnrm = get_lognorm(4, 1)
    const = constant(loc=4)
    for force in [True, False]:
        loss_expo = loss_function(expo, force=force)
        loss_gm1 = loss_function(gm1, force=force)
        loss_gm2 = loss_function(gm2, force=force)
        loss_norm = loss_function(nrm, force=force)
        loss_lnrm = loss_function(lnrm, force=force)
        loss_const = loss_function(const, force=force)
        assert 4 == pytest.approx(loss_expo(0))
        assert 4 == pytest.approx(loss_gm1(0))
        assert 4 == pytest.approx(loss_lnrm(0))
        assert 4 == pytest.approx(loss_const(0))
        x = np.array([-2, -1, 0, 1, 3, 1000])
        # For positive vars loss1(0) = mean
        loss_expo_at_x = np.array(
            [6.00000000e000, 5.00000000e000, 4.00000000e000, 3.11520313e000, 1.88946621e000, 1.06882573e-108]
        )
        np.testing.assert_allclose(loss_expo(x), loss_expo_at_x, atol=EPS, err_msg=f"Expo loss, force={force}")
        np.testing.assert_allclose(loss_gm1(x), loss_expo_at_x, atol=EPS, err_msg=f"Gamma1 loss, force={force}")
        loss_gamma2_at_x = np.array(
            [8.00000000e000, 7.00000000e000, 6.00000000e000, 5.01571917e000, 3.31091497e000, 1.72896654e-142]
        )
        np.testing.assert_allclose(loss_gm2(x), loss_gamma2_at_x, atol=EPS)
        loss_norm_at_x = np.array([6.0, 5, 4, 3.00038215, 1.08331547, 0.0])
        assert loss_norm_at_x == pytest.approx(loss_norm(x), rel=EPS)
        loss_lnrm_at_x = np.array([6, 5, 4, 3.00000000e000, 1.05078013e000, 7.72290351e-112])
        assert loss_lnrm_at_x == pytest.approx(loss_lnrm(x), rel=EPS)
        loss_const_at_x = np.array([6.0, 5.0, 4.0, 3.0, 1.0, 0.0])
        np.testing.assert_allclose(loss_const(x), loss_const_at_x, err_msg=f"Constant loss, force={force}")


def test_ph_loss():
    phd = ph_erlang(3, 10)
    gma = stats.gamma(a=3, scale=1 / 10)
    x = np.linspace(0, 2)
    loss_gma = loss_function(gma, force=False)
    loss_gma_at_x = loss_gma(x)
    loss_ph_at_x = phd.loss1(x)
    np.testing.assert_allclose(loss_gma_at_x, loss_ph_at_x)


if __name__ == "__main__":
    test_constant()
    test_loss_function()
    test_ph_loss()
