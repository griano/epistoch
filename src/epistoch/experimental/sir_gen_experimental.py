# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:50:30 2020

@author: Germán Riaño
"""

import logging
import math

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, stats
from tqdm import tqdm

import epistoch.utils as utils
from epistoch.sir_g import EPS, _compute_array_error, _sir_deriv, compute_integral, print_error, sir_g
from epistoch.utils.stats import loss_function


def deriv2(
    t, y, beta, gam, I0, times, delta, S_guess, I_guess, survival, pdfs, loss1, dist, method,
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
    # compute_integral(n, delta, S_guess, I_guess, survival, delta_loss)
    # print(f"S'({t})={dSdt}  I'({t})={dIdt}")
    return dSdt, dIdt


def stochasticSIR2(
    population=1000,
    reproductive_factor=2.0,
    disease_time_distribution=stats.expon(scale=10),
    I0=1.0,
    R0=0.0,
    num_days=160,
    num_periods=2000,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Total population, N.
    N = population
    dist = disease_time_distribution
    # Normailize to imporve numerical stability
    I0 = I0 / N
    R0 = R0 / N
    # Everyone else, S0, is susceptible to infection initially.
    S0 = 1 - I0 - R0
    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    # A grid of time points (in days)
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
    # Integrate the SIR equations over the time grid, t.
    ret = integrate.odeint(_sir_deriv, y0, times, args=(beta, gam), tfirst=True)
    old_y = ret.T
    max_diff = 1e-3
    alpha = 0.5
    for iteration in tqdm(range(max_iterations)):
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
        ret2 = integrate.odeint(func=deriv2, y0=y0, t=times, args=args, tfirst=True)
        y = ret2.T
        S, I = y
        diff = _compute_array_error("I", I_guess, I, do_print=False)
        logger.debug(f"Iteration {iteration:d}: {diff:.4%}")
        if diff < max_diff:
            break
        old_y = alpha * y + (1 - alpha) * old_y

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times, S)(days)
    I = N * interpolate.interp1d(times, I)(days)
    R = N - S - I
    return pd.DataFrame(data={"Day": days, "S": S, "I": I, "R": R}).set_index("Day")


def test_sir_g2(num_periods=2000):
    N = 1000
    dist = utils.stats.get_gamma(10, 2)
    model1 = sir_g(population=N, infectious_time_distribution=dist, num_periods=num_periods)
    model2 = stochasticSIR2(population=N, disease_time_distribution=dist, num_periods=num_periods)
    model1.plot()
    model2.plot()
    print_error(model1, model2, N)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sir_g2()
