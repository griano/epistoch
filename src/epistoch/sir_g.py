# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:20:32 2020

@author: Germán Riaño (griano@germanriano.com)
"""


import logging
import math

import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from tqdm import tqdm

from epistoch.utils.stats import loss_function
from epistoch.utils.utils import compute_integral, get_total_infected

EPS = 1e-5


# The SIR model differential equations.
def _sir_deriv(t, y, beta, gam):
    S, I = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gam * I
    return dSdt, dIdt


def sir_classical(
    name, population, reproductive_factor, infectious_period_mean, num_days, I0=1.0, S0=None, times=None,
):
    """
    Solves a classical SIR model.

    Parameters
    ----------
    name: string
        Model name
    population: float
        total population size
    reproductive_factor: float
        basic reproductive factor(R0)
    infectious_period_mean: float
        expected value of Infectious Period Time
    I0: float
        Initial infectious population
    S0: float
        Initial susceptible population (optinal, defaults to all but I0)
    num_days: int
        number of days to run
    times: array of float
        times where the functions should be reported. Defaults to
        ``np.linspace(0, num_days, num_days + 1)``


    Returns
    -------
    dict
        Dictionary with fields:
            - name: model name
            - population: Total population
            - total_infected
            - data: data Frame with columns
                - S : Susceptible,
                - I : Infectious (Dying or recovering),
                - R : Removed (recovered + deaths),
    """
    # Note: Code based on
    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    # Total population, N.
    N = population
    # Initial number of infected and recovered individuals, I0 and R0.
    I0 = I0 / N
    S0 = S0 / N if S0 is not None else 1 - I0
    # Note that R0 = 1 - I0 - S0

    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1.0 / infectious_period_mean
    beta = reproductive_factor * gam

    # A grid of time points (in days)
    times = np.linspace(0, num_days, num_days + 1) if times is None else times

    # Initial conditions vector
    y0 = [S0, I0]
    # Integrate the SIR equations over the time grid, t.
    ret = integrate.odeint(_sir_deriv, y0, times, args=(beta, gam), tfirst=True)
    S, I = ret.T
    # De-normalize
    S = S * N
    I = I * N
    R = N - S - I
    result = dict()
    result["data"] = pd.DataFrame(data={"Day": times, "S": S, "I": I, "R": R}).set_index("Day")
    result["total_infected"] = get_total_infected(reproductive_factor)
    result["population"] = N
    result["name"] = name
    return result


def sir_g(
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
    """
    Solves a SIR-G model.

    Parameters
    ----------
    name: string
        Model name
    population: float
        total population size
    reproductive_factor: float
        basic reproductive factor(R0)
    infectious_time_distribution: scipy.stats.rv_continuous
        expected value of Infectious Period Time
    num_days: int
        number of days to run
    I0: float
        Initial infectious population
    S0: float
        Initial susceptible population (optional, defaults to all but I0)
    num_periods:int
        Number of periods to use for computations. Higher number will lead ot more precise computation.
    method: string
        Method used for the internal integral
    logger
        Logger object

    Returns
    -------
    dict
        Dictionary with fields:
            - name: model name
            - population: Total population
            - total_infected
            - data: data Frame with columns
                - S : Susceptible,
                - I : Infectious (Dying or recovering),
                - R : Removed (recovered + deaths),
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    # Total population, N.
    N = population
    dist = infectious_time_distribution

    I0 = I0 / N
    S0 = S0 / N if S0 is not None else 1 - I0
    # Notice that R0 = 1 - I0 - S0

    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    num_periods = 10 * num_days if num_periods is None else num_periods
    delta = num_days / num_periods
    times = np.linspace(start=0.0, stop=num_days, num=num_periods + 1)
    logging.info("Computing survival function")
    survival = dist.sf(times)
    logging.info("Computing PDF function")
    pdfs = dist.pdf(times)
    if pdfs[0] == math.inf:
        pdfs[0] = dist.pdf(dist.ppf(0.01))
    survival = dist.sf(times)
    if survival[0] < 1.0 - EPS:
        logging.warning(f"Survival function does not start in 1:  {survival}")
    logging.info("Computing Loss1 function")
    loss_fun = loss_function(dist)
    loss1 = loss_fun(times)

    S = np.zeros(num_periods + 1)
    # S[:] = np.nan
    I = np.zeros(num_periods + 1)
    #    I[:] = np.nan
    S[0] = S0
    I[0] = I0
    logging.info(f"Computing SIR-G model with {num_periods} periods.")
    for n in tqdm(range(0, num_periods)):
        S[n + 1] = S[n] - delta * beta * S[n] * I[n]
        integral = compute_integral(n, delta, S, I, times, survival, pdfs, loss1, dist, method=method)
        # Expected case for exponential variables
        integral_expon = (gam / beta) * (I[n] - I0 * survival[n])
        logger.debug(f"n={n}, t={n*delta}, integral = {integral}, integral_expon = {integral_expon}")
        I[n + 1] = I[n] + delta * (beta * (S[n] * I[n] - integral) - gam * I0 * survival[n])

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times, S)(days)
    I = N * interpolate.interp1d(times, I)(days)
    R = N - S - I
    result = dict()
    result["name"] = name
    result["population"] = population
    result["data"] = pd.DataFrame(data={"Day": days, "S": S, "I": I, "R": R}).set_index("Day")
    result["total_infected"] = get_total_infected(reproductive_factor)
    return result
