# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:50:30 2020

@author: Germán Riaño
"""

import logging
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate, interpolate, stats
from tqdm import tqdm

import epistoch.utils as utils
from epistoch import compute_integral, get_total_infected, sir_classical
from epistoch.sir_g import EPS, _sir_deriv, sir_g
from epistoch.utils.plotting import plot_sir
from epistoch.utils.stats import loss_function
from epistoch.utils.utils import _compute_array_error, print_error


def _deriv2(
    # Derivatives for second step. S and I are fixed.
    t,
    y,
    beta,
    gam,
    I0,
    times,
    delta,
    S_guess,
    I_guess,
    survival,
    pdfs,
    loss1,
    dist,
    method,
):
    # This function does not depend on y, but it depends on t
    # We buid piece-wise linear functions
    if t < min(times) or t > max(times):
        return 0.0, 0.0

    S = interpolate.interp1d(times, S_guess, fill_value="extrapolate")
    I = interpolate.interp1d(times, I_guess, fill_value="extrapolate")
    dSdt = -beta * S(t) * I(t)
    n = int(np.floor(t / delta))
    integral = compute_integral(n, delta, S_guess, I_guess, times, survival, pdfs, loss1, dist, method)
    dIdt = -dSdt - beta * integral - gam * I0 * survival[n]

    return dSdt, dIdt


def sir_g2(
    name,
    population,
    reproductive_factor,
    infectious_time_distribution,
    num_days,
    I0=1.0,
    S0=None,
    num_periods=None,
    method="loss",
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Total population, N.
    N = population
    dist = infectious_time_distribution
    # Normailize to imporve numerical stability
    I0 = I0 / N
    S0 = S0 / N if S0 is not None else 1 - I0
    # Notice that R0 = 1 - I0 - S0

    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    num_periods = 10 * num_days if num_periods is None else num_periods
    delta = num_days / num_periods
    times = np.linspace(start=0.0, stop=num_days, num=num_periods + 1)
    survival = dist.sf(times)
    pdfs = dist.pdf(times)
    if pdfs[0] == math.inf:
        pdfs[0] = dist.pdf(dist.ppf(0.01))
    survival = dist.sf(times)
    if survival[0] < 1.0 - EPS:
        logging.warning(f"Survival function does not start in 1:  {survival}")
    loss_fun = loss_function(dist)
    loss1 = loss_fun(times)

    S = np.empty(num_periods + 1)
    S[:] = np.nan
    I = np.empty(num_periods + 1)
    I[:] = np.nan
    S[0] = S0
    I[0] = I0

    # Initial conditions vector
    y0 = [S0, I0]

    max_iterations = 100
    logger.info("Computing classical SIR")
    sir_class = sir_classical(
        name="Classic",
        population=1,
        reproductive_factor=reproductive_factor,
        infectious_period_mean=dist.mean(),
        num_days=num_days,
        I0=I0,
        times=times,
    )
    data = sir_class["data"]
    old_y = np.asarray((np.asarray(data.S), np.asarray(data.I)))
    # old_y = np.asarray((.5*np.ones_like (times), .5* np.ones_like (times)))
    # old_y = np.asarray((np.zeros_like (times), np.zeros_like (times)))
    max_diff = 1e-6
    alpha = 0.9
    for iteration in range(max_iterations):
        S_guess, I_guess = old_y
        # Integrate again, with the integral, fixing guess
        args = (
            beta,
            gam,
            I0,
            times,
            delta,
            S_guess,
            I_guess,
            survival,
            pdfs,
            loss1,
            dist,
            "loss",
        )
        logger.debug(f"Staring iteration {iteration:d}")
        ret2 = integrate.odeint(func=_deriv2, y0=y0, t=times, args=args, tfirst=True)
        y = ret2.T
        S, I = y
        diff = _compute_array_error("I", I_guess, I, N, do_print=False)
        if logger.isEnabledFor(logging.DEBUG):
            pd.DataFrame(data={"Times": times, "S": S, "I": I}).set_index("Times").plot()
            plt.show()
        logger.debug(f"Iteration {iteration:d}: {diff:.4%}")
        if diff < max_diff:
            break
        old_y = alpha * y + (1 - alpha) * old_y

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times, S)(days)
    I = N * interpolate.interp1d(times, I)(days)
    R = N - S - I
    result = dict()
    result["data"] = pd.DataFrame(data={"Day": days, "S": S, "I": I, "R": R}).set_index("Day")
    result["total_infected"] = get_total_infected(reproductive_factor)
    result["population"] = N
    result["name"] = name
    return result


def compare_sir_g2(dist, num_periods=2000, logger=None):
    N = 1000
    reproductive_factor = 2.2
    num_days = 160
    model0 = sir_classical(
        "SIR_CL",
        population=N,
        reproductive_factor=reproductive_factor,
        infectious_period_mean=dist.mean(),
        num_days=num_days,
    )
    model1 = sir_g(
        "SIR_G",
        population=N,
        reproductive_factor=reproductive_factor,
        infectious_time_distribution=dist,
        num_days=num_days,
        num_periods=num_periods,
        logger=None,
    )
    model2 = sir_g2(
        "SIR_G2",
        population=N,
        reproductive_factor=reproductive_factor,
        infectious_time_distribution=dist,
        num_days=num_days,
        num_periods=num_periods,
        logger=logger,
    )
    print_error(model1, model2)
    fig = plot_sir(model1)
    plot_sir(model2, fig=fig, linestyle="--")
    plot_sir(model0, fig=fig, linestyle=":")
    plt.show()


def test_sir_g2(dist=utils.stats.get_gamma(10, 2), num_periods=2000, logger=None):
    compare_sir_g2(dist, num_periods, logger)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    test_sir_g2(dist=utils.stats.get_gamma(10, 20), num_periods=2000, logger=logger)
