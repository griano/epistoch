# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:43:11 2020

@author: Germán Riaño, PhD
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy import matlib as ml
from scipy import integrate, interpolate, linalg, stats
from tqdm import tqdm

from epi_stoch.SIR_general import classicalSIR, print_error, stochasticSIR
from epi_stoch.utils.plotting import plot_IR, plot_sir
from epi_stoch.utils.stats import loss_function
from pyphase.ph import ph_erlang, ph_expon

EPS = 1e-5


def to_array(S, x):
    S = np.array(S).flatten()
    x = np.array(x).flatten()
    return np.concatenate((S, x))

# The SIR-PH model differential equations.
def deriv(t, y, beta, n, alpha, A, a):
    S, x = np.split(y, [1])
    x = ml.mat(x)
    x.reshape(1,n)
    x_beta = (x @ beta)
    dSdt = - S * x_beta
    dxdt =   S * (x @ beta @ alpha) + x @ A
    return to_array(dSdt, dxdt)

def sir_phg(population=1000, 
            beta = 0.2,
            disease_time_distribution=ph_expon(lambd=1/10), 
            I0 = 1.0, 
            R0 = 0.0,
            num_days = 160,
            logger=None,
            report_phases=False):
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
    # A grid of time points (in days)
    times = np.linspace(start=0.0, stop=num_days, num=num_days+1)
    
    alpha, A, a, n = dist.params()
    beta = beta * np.ones(n)

    x0 = -I0 * gam * alpha @ linalg.inv(A)
    
    logging.info(f"Computing SIR-PH model with {num_days} days and {n} phases.")
    # Initial conditions vector
    y0 = to_array(S0, x0)
    # Integrate the SIR equations over the time grid, t.
    ret = integrate.odeint(deriv, y0, times, args=(beta, n, alpha, A, a), tfirst=True)
    S, x = np.split(ret, [1], axis=1)
 
    S = S.flatten()
    I = np.sum(x, axis=1) 

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times,S)(days)
    I = N * interpolate.interp1d(times,I)(days)
    R = N - S - I
    return pd.DataFrame(data={'Day':days, 'S': S, 'I':I,'R': R}).set_index('Day')



###########################
# TESTING. To be moved
##########################

def test_sir_phg_expon(do_plot=False):
    N = 1000
    reproductive_factor = 2.2
    infectious_period_mean =10
    dist = ph_expon(lambd=1/infectious_period_mean)
    assert(dist.mean() == infectious_period_mean)
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    sir = classicalSIR(N, reproductive_factor=reproductive_factor)
    sir_ph_exp = sir_phg(N, beta=beta, disease_time_distribution=dist)
    error = print_error(sir, sir_ph_exp, N)
    if do_plot:
        fig = plot_sir('SIR', sir, N, title='PH-expo-test') 
        plot_sir('SIR-PHG', sir_ph_exp, N, fig, linestyle='--')    
        plt.show()
        assert(0.0 == pytest.approx(error, abs=1e-2))    

def test_sir_phg_erlang(do_plot = False):
    N = 1000
    reproductive_factor = 2.2
    infectious_period_mean = 10
    n = 3
    dist = ph_erlang(n=n, lambd=n/infectious_period_mean)
    assert(dist.mean() == infectious_period_mean)
    gam = 1 / dist.mean()
    beta = reproductive_factor * gam
    sir_g = stochasticSIR(N, reproductive_factor=reproductive_factor, disease_time_distribution=dist)
    sir_ph_erlang = sir_phg(N, beta=beta, disease_time_distribution=dist)
    error = print_error(sir_g,sir_ph_erlang, N)
    if do_plot:
        fig = plot_sir('SIR-G', sir_g, N, title='PH-erlang-test') 
        plot_sir('SIR-PHG', sir_ph_erlang, N, fig, linestyle='--')    
        plt.show()
    # assert(0.0 == pytest.approx(error, abs=1e-2))    
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_sir_phg_expon()
    test_sir_phg_erlang(True)
