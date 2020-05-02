# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:34:26 2020

@author: Germán Riaño
"""

import logging
import os
from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy

import epistoch.utils as utils
from epistoch.sir_g import print_error, report_summary, sir_classical, sir_g
from epistoch.utils.plotting import plot_IR, plot_sir


def performance_test(N):
    tt = timeit(
        setup="from __main__ import classicalSIR",
        stmt="tc, Sc, Ic, Rc = classicalSIR(population=1000, use_odeint=True)",
        number=1000,
    )
    print(f"tt odeint = {tt}")
    tt = timeit(
        setup="from __main__ import classicalSIR",
        stmt="tc, Sc, Ic, Rc = classicalSIR(population=1000, use_odeint=False)",
        number=1000,
    )
    print(f"tt ivp = {tt}")
    tc, Sc, Ic, Rc = sir_classical(population=N, use_odeint=True)
    tc2, Sc2, Ic2, Rc2 = sir_classical(population=N, use_odeint=False)
    print(np.max(np.abs(Ic - Ic2)) / N)
    print(np.max(np.abs(Sc - Sc2)) / N)


def compare_models(name, dist, N=1000000, I0=1, num_days=100, R0=2.25, do_plots=True):
    name1 = "SIR"
    SIR_classic = sir_classical(
        population=N, I0=I0, reproductive_factor=R0, infectious_period_mean=dist.mean(), num_days=num_days,
    )

    name2 = "SIR-G"
    SIR_general = sir_g(
        population=N,
        I0=I0,
        num_days=num_days,
        num_periods=2000,
        reproductive_factor=R0,
        infectious_time_distribution=dist,
        method="loss",
    )
    report_summary(name1, SIR_classic, N)
    report_summary(name2, SIR_general, N)
    print_error(SIR_classic, SIR_general, N)

    if do_plots:
        fig = plot_sir(name1, SIR_classic, N, title=name + ": SIR and SIR-G models")
        plot_sir(name2, SIR_general, N, fig, linestyle="--")
        plt.show()
        filename = os.path.join("./paper/epistoch/figures/", name + "-SIR-comp.pdf")
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig.savefig(filename, bbox_inches="tight")

        fig2 = plot_IR(name1, SIR_classic, N, title=name + ": I, R as function of S")
        plot_IR(name2, SIR_general, N, fig2, linestyle="--")
        plt.show()
        filename = os.path.join("./paper/epistoch/figures/", name + "-IR-comp.pdf")
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig2.savefig(filename, bbox_inches="tight")
        print("Done")


def variance_analysis(
    N=1000000, I0=1, num_days=100, R0=2.25, infectious_period_mean=4.4, cvs=[0.5, 1.0, 2.0],
):

    dists = {}
    models = {}
    print("Computing Classical")
    sir_classic = sir_classical(
        population=N, I0=I0, reproductive_factor=R0, infectious_period_mean=infectious_period_mean, num_days=num_days,
    )
    fig = plot_sir("SIR", sir_classic, N, formats={"I": "r-"}, title="SIR-G with Different Distributions", linewidth=3,)

    print("Computing Constant SIR-G")
    dist = utils.stats.constant(infectious_period_mean)
    sir_constant = sir_g(
        population=N,
        I0=I0,
        num_days=num_days,
        num_periods=2000,
        reproductive_factor=R0,
        infectious_time_distribution=dist,
        method="loss",
    )
    fig = plot_sir("Const", sir_constant, N, fig, formats={"I": "b-."}, linewidth=3)

    dists["gamma"] = utils.stats.get_gamma
    dists["lognorm"] = utils.stats.get_lognorm

    for cv in cvs:
        for dist_name, dist_getter in dists.items():
            infectious_period_sd = cv * infectious_period_mean
            dist = dist_getter(infectious_period_mean, infectious_period_sd)
            mod_name = f"{dist_name}-{cv:.2}"
            print(f"Running {mod_name}")
            num_days = round(100 * max(1, cv))
            models[(dist, cv)] = sir_g(
                population=N,
                I0=I0,
                num_days=num_days,
                num_periods=2000,
                reproductive_factor=R0,
                infectious_time_distribution=dist,
                method="loss",
            )
            #            models[(dist, cv)].name = mod_name
            plot_sir(mod_name, models[(dist, cv)], N, fig, formats={"I": ""}, linestyle="--")
    plt.show()

    filename = os.path.join("./paper/epistoch/figures/", "Variance-Analysis.pdf")
    print(f"Saving picture in file {os.path.abspath(filename)}")
    fig.savefig(filename, bbox_inches="tight")

    return models


if __name__ == "__main__":
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.basicConfig()
    compare_models("DIVOC", scipy.stats.norm(loc=4.5, scale=1))
    variance_analysis()
