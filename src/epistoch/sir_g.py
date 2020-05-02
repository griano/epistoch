# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:20:32 2020

@author: Germán Riaño (griano@germanriano.com)
"""


import logging
import math

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, stats
from tqdm import tqdm

from epistoch.utils.stats import loss_function

EPS = 1e-5


# The SIR model differential equations.
def deriv(t, y, beta, gam):
    S, I = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gam * I
    return dSdt, dIdt


def sir_classical(
    population=1000, reproductive_factor=2.0, infectious_period_mean=10, I0=1.0, S0=None, num_days=160, use_odeint=True,
):
    """
    Solves a classical SIR model
    Parameters
    ----------
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
    use_odeint: bool


    Returns
    -------
    A dictionary with these fields:
        data: DataFrame with columns S, I, R
        total_infected: estimation of total infected individuals

    Note: Code based on
    https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    """
    # Total population, N.
    N = population
    # Initial number of infected and recovered individuals, I0 and R0.
    I0 = I0 / N
    S0 = S0 / N if S0 is not None else 1 - I0
    R0 = 1 - I0 - S0

    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1.0 / infectious_period_mean
    beta = reproductive_factor * gam

    # A grid of time points (in days)
    times = np.linspace(0, num_days, num_days + 1)

    # Initial conditions vector
    y0 = [S0, I0]
    # Integrate the SIR equations over the time grid, t.
    if use_odeint:
        ret = integrate.odeint(deriv, y0, times, args=(beta, gam), tfirst=True)
        S, I = ret.T
    else:
        sol = integrate.solve_ivp(
            fun=deriv, t_span=[0, num_days], y0=y0, args=(beta, gam), t_eval=times, vectorized=True,
        )
        times = sol.t
        S, I = sol.y
    S = S * N
    I = I * N
    R = N - S - I
    result = dict()
    result["data"] = pd.DataFrame(data={"Day": times, "S": S, "I": I, "R": R}).set_index("Day")
    result["total_infected"] = get_total_infected(reproductive_factor)
    return result


def compute_integral(n, delta, S, I, times, survival, pdfs, loss1, dist, method="loss"):
    """
    Computes
    int_0^t g(t-x) I(x)S(x) dx
    for t = n*delta


    Parameters
    ----------
    n : integer
        upper limit for integral.
    delta : float
        interval size
    survival : array of float
        array G_k = P( T > delta k)
    delta_loss : array of float
        array L_k = E(T-delta*k)

    Returns
    -------
    Integral value

    """
    if n == 0:
        return 0.0
    if method == "loss":
        IS = np.zeros_like(survival)
        # next line equivalent to: IS[: n + 1] = np.array([I[n - k] * S[n - k] for k in range(n + 1)])
        IS[: n + 1] = np.flipud(I[: n + 1] * S[: n + 1])
        slopes = np.diff(IS, append=0.0) / delta  # m1, m2, ...
        delta_slopes = np.diff(slopes, prepend=0.0)
        return IS[0] + sum(delta_slopes * loss1)

    if method == "simpson":
        integral_points = [pdfs[n - k] * S[k] * I[k] for k in range(0, n + 1)]
        return integrate.simps(integral_points, dx=delta)
    if method == "interpolate":
        t = n * delta
        interpolator = interpolate.interp1d(times[: n + 1], S[: n + 1] * I[: n + 1])

        def integrand(tau):
            return dist.pdf(t - tau) * interpolator(tau)

        return integrate.quad(integrand, 0, t)[0]


def sir_g(
    population=1000,
    reproductive_factor=2.0,
    infectious_time_distribution=stats.expon(scale=10),
    I0=1.0,
    S0=None,
    num_days=160,
    num_periods=None,
    method="loss",
    logger=None,
):
    """
    Solves a SIR-G model
    Parameters
    ----------
    population: float
        total population size
    reproductive_factor: float
        basic reproductive factor(R0)
    infectious_time_distribution: scipy.stats.rv_continuous
        expected value of Infectious Period Time
    I0: float
        Initial infectious population
    S0: float
        Initial susceptible population (optinal, defaults to all but I0)
    num_days: int
        number of days to run
    num_periods:int
        Number of periods to use for computations. Higher number will lead ot more precise computation.
    method: string
        Method used for the internal integral
    logger
        Logger object
    Returns
    -------
    A dictionary with these fields:
        data: DataFrame with columns S, I, R
        total_infected: estimation of total infected individuals

    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    # Total population, N.
    N = population
    dist = infectious_time_distribution

    I0 = I0 / N
    S0 = S0 / N if S0 is not None else 1 - I0
    R0 = 1 - I0 - S0

    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    # A grid of time points (in days)
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
    result["data"] = pd.DataFrame(data={"Day": days, "S": S, "I": I, "R": R}).set_index("Day")
    result["total_infected"] = get_total_infected(reproductive_factor)
    return result


def get_total_infected(reproductive_factor, normalzed_s0=1):
    if reproductive_factor < 1:
        return 0.0
    else:
        fun = lambda x: 1 - x - normalzed_s0 * np.exp(-reproductive_factor * x)
        result = optimize.root_scalar(fun, bracket=[0.0001, 1])
        return result.root


def get_array_error(name, x1, x2, N=1, do_print=True):
    error = np.abs(x1 - x2) / N
    if do_print:
        print(f"{name}: max error = {np.max(error):.2}, avg error = {np.mean(error):.2}")
    return np.max(error)


def get_error(model1, model2, N, do_print=True):
    error_i = get_array_error("I", model1["data"].I, model2["data"].I, N, do_print)
    error_s = get_array_error("S", model1["data"].S, model2["data"].S, N, do_print)
    return 0.5 * (error_i + error_s)


def print_error(model1, model2, N):
    return get_error(model1, model2, N, True)


def report_summary(name, result, N):
    model = result["data"]
    n = len(model) - 1
    print(f"Model {name} Summary")
    print(f"  Total Infected People: {int(model.R[n]):,d} ({model.R[n]/N:.2%})")
    print(f"  Infection Peak: {int(np.max(model.I)):,d} ({np.max(model.I)/N:.2%})")
    print(f"  Peak Day: {int(np.argmax(model.I)):,d}")
    if "total_infected" in result:
        print(f"  Theoretical Total Infected: {result['total_infected']:.2%}")
