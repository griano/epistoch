# -*- coding: utf-8 -*-

import logging
import os

import matplotlib.pyplot as plt
import scipy

import epistoch.utils as utils
from epistoch.sir_g import sir_classical, sir_g
from epistoch.utils.plotting import plot_IR, plot_sir
from epistoch.utils.utils import print_error, report_summary


def compare_models(
    name, dist, N, I0=1, num_days=100, R0=2.25, do_plots=True, use_latex=False, plot_path=".", plot_ext="pdf"
):
    name1 = "SIR"
    SIR_classic = sir_classical(
        name=name1, population=N, I0=I0, reproductive_factor=R0, infectious_period_mean=dist.mean(), num_days=num_days,
    )

    name2 = "SIR-G"
    SIR_general = sir_g(
        name=name2,
        population=N,
        I0=I0,
        num_days=num_days,
        num_periods=2000,
        reproductive_factor=R0,
        infectious_time_distribution=dist,
        method="loss",
    )
    report_summary(SIR_classic)
    report_summary(SIR_general)
    print_error(SIR_classic, SIR_general)

    if do_plots:
        fig = plot_sir(SIR_classic, title=name + ": SIR and SIR-G models", use_latex=use_latex)
        plot_sir(SIR_general, fig, linestyle="--")
        plt.show()
        filename = os.path.join(plot_path, name + "-SIR-comp." + plot_ext)
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig.savefig(filename, bbox_inches="tight")

        fig2 = plot_IR(SIR_classic, title=name + ": I, R as function of S")
        plot_IR(SIR_general, fig2, linestyle="--")
        plt.show()
        filename = os.path.join(plot_path, name + "-IR-comp." + plot_ext)
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig2.savefig(filename, bbox_inches="tight")
        print("Done")


def variance_analysis(
    N=1000000,
    I0=1,
    num_days=100,
    R0=2.25,
    infectious_period_mean=4.4,
    cvs=[0.5, 1.0, 2.0],
    use_latex=False,
    plot_path=".",
    plot_ext="pdf",
):

    dists = {}
    models = {}
    print("Computing Classical")
    sir_classic = sir_classical(
        name="SIR",
        population=N,
        I0=I0,
        reproductive_factor=R0,
        infectious_period_mean=infectious_period_mean,
        num_days=num_days,
    )
    fig = plot_sir(sir_classic, formats={"I": "r-"}, title="SIR-G with Different Distributions", linewidth=3,)

    print("Computing Constant SIR-G")
    dist = utils.stats.constant(infectious_period_mean)
    sir_constant = sir_g(
        name="Const",
        population=N,
        I0=I0,
        num_days=num_days,
        num_periods=2000,
        reproductive_factor=R0,
        infectious_time_distribution=dist,
        method="loss",
    )
    fig = plot_sir(sir_constant, fig, formats={"I": "b-."}, linewidth=3, use_latex=use_latex)

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
                name=mod_name,
                population=N,
                I0=I0,
                num_days=num_days,
                num_periods=2000,
                reproductive_factor=R0,
                infectious_time_distribution=dist,
                method="loss",
            )
            #            models[(dist, cv)].name = mod_name
            plot_sir(models[(dist, cv)], fig, formats={"I": ""}, linestyle="--")
    plt.show()

    filename = os.path.join(plot_path, "Variance-Analysis." + plot_ext)
    print(f"Saving picture in file {os.path.abspath(filename)}")
    fig.savefig(filename, bbox_inches="tight")

    return models


if __name__ == "__main__":
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    plot_path = "./paper/epistoch/figures/"
    compare_models(
        "DIVOC",
        N=1000000,
        I0=1,
        num_days=100,
        R0=2.25,
        dist=scipy.stats.norm(loc=4.5, scale=1),
        use_latex=True,
        plot_path=plot_path,
    )
    variance_analysis(
        N=1000000,
        I0=1,
        num_days=100,
        R0=2.25,
        infectious_period_mean=4.5,
        cvs=[0.5, 1.0, 2.0],
        use_latex=True,
        plot_path=plot_path,
    )
