# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:43:11 2020

@author: Germán Riaño, PhD
"""
import logging

import numpy as np
import pandas as pd
from numpy import matlib as ml
from scipy import integrate, interpolate, linalg

from epistoch.sir_g import get_total_infected
from pyphase.phase import ph_expon

EPS = 1e-5


def to_array(*args):
    flat_args = [np.array(arg).flatten() for arg in args]
    return np.concatenate(flat_args)


# The SIR-PH model differential equations.
def deriv(t, y, beta, n, alpha, A, a):
    S, x, r = np.split(y, [1, n + 1])
    x = ml.mat(x)
    x.reshape(1, n)
    x_beta = x @ beta
    dSdt = -S * x_beta
    dxdt = S * (x @ beta @ alpha) + x @ A
    drdt = np.multiply(x.T, a)  # component multiplication
    return to_array(dSdt, dxdt, drdt)


def sir_phg(
    population=1000,
    beta=0.2,
    infectious_time_distribution=ph_expon(lambd=1 / 10),
    I0=1.0,
    S0=None,
    num_days=160,
    logger=None,
    report_phases=False,
):
    """
    Compute a SIR-PH model

    Parameters
    ----------
    population : float, optional
        Total population. The default is 1000.
    beta : float, optional
        Contagion rate. The default is 0.2.
    infectious_time_distribution : phase, optional
        Must be a PH dstribution. The default is ph_expon(lambd=1 / 10).
    I0 : float, optional
        initial population. The default is 1.0.
    S0 : TYPE, optional
        Initial susceptible. The default is all but I0.
    num_days : int, optional
        Number of days to run. The default is 160.
    logger : TYPE, optional
        Logger object. The default is None.
    report_phases : TYPE, optional
        Wether to include phase levels in the report. The default is False.

    Returns
    -------
    A dictionary with these fields:
        data: DataFrame with columns S, I, R. Optionally: I-Phase0,...,I-Phase(n-1), and R-Phase0,...R-Phase(n-1)
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
    # A grid of time points (in days)
    times = np.linspace(start=0.0, stop=num_days, num=num_days + 1)

    alpha, A, a, n = dist.params()
    beta = beta * np.ones(n)

    # Spread initial I in phases
    x0 = -I0 * gam * alpha @ linalg.inv(A)
    # Spread initial R0 evenly  in phases
    r0 = R0 * np.ones(n) / n

    logging.info(f"Computing SIR-PH model with {num_days} days and {n} phases.")
    # Initial conditions vector
    y0 = to_array(S0, x0, r0)
    # Integrate the SIR-PH equations over the time grid, t.
    ret = integrate.odeint(deriv, y0, times, args=(beta, n, alpha, A, a), tfirst=True)
    S, x, r = np.split(ret, [1, n + 1], axis=1)

    S = S.flatten()
    I = np.sum(x, axis=1)
    R = np.sum(r, axis=1)
    pi = dist.equilibrium_pi()
    effective_r0 = (pi @ beta) / gam

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times, S)(days)
    I = N * interpolate.interp1d(times, I)(days)
    R = N * interpolate.interp1d(times, R)(days)
    result = dict()
    result["data"] = pd.DataFrame(data={"Day": days, "S": S, "I": I, "R": R}).set_index("Day")
    if report_phases:
        xdf = pd.DataFrame(N * x)
        xdf.columns = ["I-Phase" + str(i) for i in range(n)]
        result["data"] = pd.concat([result["data"], xdf], axis=1)
        result["I-columns"] = xdf.columns.values.tolist()
        rdf = pd.DataFrame(N * r)
        rdf.columns = ["R-Phase" + str(i) for i in range(n)]
        result["data"] = pd.concat([result["data"], rdf], axis=1)
        result["R-columns"] = rdf.columns.values.tolist()
    result["total_infected"] = get_total_infected(effective_r0, S0)
    return result
