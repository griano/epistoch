import logging
import os

import pytest
from matplotlib import pyplot as plt
from scipy import stats

from epistoch.seird_ph import seird_ph
from epistoch.sir_g import sir_classical, sir_g
from epistoch.sir_phg import sir_phg
from epistoch.utils.plotting import plot_sir
from epistoch.utils.utils import print_error, report_summary
from pyphase.phase import ph_erlang, ph_expon


def test_sir_g():
    population = 1000
    reproductive_factor = 2.0
    infectious_period_mean = 10
    num_days = 160
    sir_classic = sir_classical(
        name="SIR",
        population=population,
        reproductive_factor=reproductive_factor,
        infectious_period_mean=infectious_period_mean,
        num_days=num_days,
    )
    dist = stats.expon(scale=infectious_period_mean)
    sir_gen = sir_g(
        name="SIR-G",
        population=population,
        reproductive_factor=reproductive_factor,
        infectious_time_distribution=dist,
        num_days=num_days,
    )
    error = print_error(sir_classic, sir_gen)
    report_summary(sir_classic)
    report_summary(sir_gen)
    assert 0.0 == pytest.approx(error, abs=1e-2)


def profile_sir_g():
    import cProfile
    import pstats

    cProfile.run("stochasticSIR(N)", "restats")
    p = pstats.Stats("restats")
    return p


def test_sir_phg_expon(do_plot=False):
    N = 1000
    reproductive_factor = 2.2
    infectious_period_mean = 10
    dist = ph_expon(lambd=1 / infectious_period_mean)
    assert dist.mean() == infectious_period_mean
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    sir = sir_classical(
        name="SIR",
        population=N,
        reproductive_factor=reproductive_factor,
        infectious_period_mean=infectious_period_mean,
        num_days=160,
    )
    sir_ph_exp = sir_phg(name="SIR-PH-Expo", population=N, beta=beta, infectious_time_distribution=dist, num_days=160)
    error = print_error(sir, sir_ph_exp)
    if do_plot:
        fig = plot_sir(sir, title="PH-expo-tests")
        plot_sir(sir_ph_exp, N, fig, linestyle="--")
        plt.show()
        assert 0.0 == pytest.approx(error, abs=1e-2)


def test_sir_phg_erlang(do_plot=False):
    N = 1000
    reproductive_factor = 2.2
    infectious_period_mean = 10
    n = 3
    dist = ph_erlang(n=n, lambd=n / infectious_period_mean)
    assert dist.mean() == infectious_period_mean
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    num_days = 160
    sir_g_model = sir_g(
        "SIR-G",
        population=N,
        reproductive_factor=reproductive_factor,
        infectious_time_distribution=dist,
        num_days=num_days,
        num_periods=2000,
    )
    sir_ph_erlang = sir_phg("SIR-PHG", population=N, beta=beta, infectious_time_distribution=dist)
    error = print_error(sir_g_model, sir_ph_erlang)
    if do_plot:
        fig = plot_sir(sir_g_model, title="PH-erlang-tests")
        plot_sir(sir_ph_erlang, fig, linestyle="--")
        plt.show()
    assert 0.0 == pytest.approx(error, abs=1e-2)


def test_seird(do_plot=False):
    N = 1000
    beta = 0.2

    exposed_time_mean = 1
    die_time_mean = 7
    recover_time_mean = 12

    exposed_time_exp = ph_expon(mean=exposed_time_mean)
    die_time_exp = ph_expon(mean=die_time_mean)
    recover_time_exp = ph_expon(mean=recover_time_mean)

    exposed_time = ph_erlang(n=10, mean=exposed_time_mean)
    die_time = ph_erlang(n=15, mean=die_time_mean)
    recover_time = ph_erlang(n=8, mean=recover_time_mean)

    fatality_rate = 0.2
    I0 = 10
    num_days = 200

    model_expo = seird_ph(
        "SEIRD-Expo",
        population=N,
        beta=beta,
        exposed_time=exposed_time_exp,
        time_to_die=die_time_exp,
        time_to_recover=recover_time_exp,
        fatality_rate=fatality_rate,
        I0=I0,
        num_days=num_days,
    )
    model_gen = seird_ph(
        "SEIRD-Gen",
        population=N,
        beta=beta,
        exposed_time=exposed_time,
        time_to_die=die_time,
        time_to_recover=recover_time,
        fatality_rate=fatality_rate,
        I0=I0,
        num_days=num_days,
    )
    if do_plot:
        formats = {"S": "b-", "E": "c-", "I": "r-", "R": "g-", "Rc": "y-", "D": "m-"}
        legend_fmt = {"loc": "upper right", "shadow": True, "framealpha": 1.0, "bbox_to_anchor": (1, 1)}
        report_summary(model_expo)
        fig = plot_sir(model_expo, title="SEIRD Expo vs Erlang", formats=formats)

        report_summary(model_gen)
        fig = plot_sir(model_gen, formats=formats, fig=fig, linestyle=":", legend_fmt=legend_fmt)

        plt.show()
        filename = os.path.join("./paper/epistoch/figures/", "SEIRD.pdf")
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_sir_g()

    test_sir_phg_expon()
    test_sir_phg_erlang(True)

    test_seird(do_plot=True)
