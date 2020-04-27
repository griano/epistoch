# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:50:30 2020

@author: germanr
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, stats
from tqdm import tqdm

import epi_stoch.utils as utils
from epi_stoch.SIR_general import (
    EPS,
    classicalSIR,
    compute_integral,
    deriv,
    get_array_error,
    print_error,
    stochasticSIR,
)
from epi_stoch.utils.stats import loss_function


def deriv2(t, y, beta, gam, I0, times, delta, S_guess, I_guess, survival, pdfs, loss1, dist, method):
    # This function does not depend on y, but it depends on t
    # We buid piece-wise linear functions
    if t < min(times) or t > max(times):
        return 0., 0.
        
    S = interpolate.interp1d(times, S_guess, fill_value='extrapolate')
    I = interpolate.interp1d(times, I_guess, fill_value='extrapolate')
    dSdt = -beta * S(t) * I(t)
    n = int(np.floor(t /delta))
    integral = compute_integral(n, delta, S_guess, I_guess, times, survival, pdfs, loss1, dist, method)
    dIdt = -dSdt - beta * integral - gam * I0 * survival[n]
    # compute_integral(n, delta, S_guess, I_guess, survival, delta_loss)
    # print(f"S'({t})={dSdt}  I'({t})={dIdt}")
    return dSdt, dIdt


def stochasticSIR2(population=1000,
                   reproductive_factor=2.0,
                   disease_time_distribution=stats.expon(scale=10),
                   I0 = 1.0,
                   R0 = 0.0,
                   num_days = 160,
                   delta = 1.,
                   logger=None):
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
    num_periods = round (num_days/delta)
    times = np.linspace(start=0.0, stop=num_days, num=num_periods+1)
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
    ret = integrate.odeint(deriv, y0, times, args=(beta, gam), tfirst=True)
    old_y = ret.T
    max_diff = 1e-3
    alpha = .5
    for iteration in tqdm(range(max_iterations)):
        S_guess, I_guess = old_y
       # Integrate again, with the integral, fixing guess 
        args = (beta, gam, I0, times, delta, S_guess, I_guess, survival, pdfs, loss1, dist, 'loss')
        ret2 = integrate.odeint(func=deriv2, y0=y0, t=times, args=args, tfirst=True)
        y = ret2.T
        S, I = y
        diff = get_array_error('I', I_guess, I, do_print=False)
        logger.debug(f'Iteration {iteration:d}: {diff:.4%}')
        if diff < max_diff:
            break
        old_y = alpha * y + (1-alpha) * old_y
        
            

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times,S)(days)
    I = N * interpolate.interp1d(times,I)(days)
    R = N - S - I
    return pd.DataFrame(data={'Day':days, 'S': S, 'I':I,'R': R}).set_index('Day')








def residual(x, S0, I0, beta, num_periods, delta, dist):
    S, I = np.split(x, 2)
    SI  = S*I
    
    dSdt = np.zeros_like(S)
    dIdt = np.zeros_like(I)
    integral = np.zeros_like(I)
    dSdt[1:] = (S[1:] - S[0:-1]) / delta
    dSdt[0]  = (S[0] - S0) /delta
    dIdt[1:] = (I[1:] - I[0:-1]) / delta
    dIdt[0]  = (I[0] - I0) /delta

    for n in range(num_periods):
        t = delta*(n+1)
        integrand = [dist.pdf(t-k*delta)  for k in range(n+1) ] * SI[:n+1]
        integral[n] = integrate.simps(integrand, range(n+1))
    residual_S = dSdt - beta * S * I 
    residual_I = dIdt - (beta * S * I  - beta * integral)
    return np.concatenate((residual_S, residual_I))

def stochasticSIR3(population=1000,
                   reproductive_factor=2.0,
                   disease_time_distribution=stats.expon(scale=10),
                   I0 = 1.0,
                   R0 = 0.0,
                   num_days = 160,
                   delta = 1.):
    # Total population, N.
    N = population
    dist = disease_time_distribution
    # Initial number of infected and recovered individuals, I0 and R0.
    
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta = reproductive_factor / dist.mean() 
    betaN = beta/N
    # A grid of time points (in days)
    num_periods = round (num_days/delta)
    
    times, S, I, R = classicalSIR(population=N,
                 reproductive_factor=reproductive_factor,
                 mean_disease_time=dist.mean(),
                 I0 = I0,
                 R0 = R0,
                 num_days = num_days)
    
    # Use as Guess the classical solution
    x0 = np.concatenate([S, I])
    funct = lambda x: residual(x, S0, I0, betaN, num_periods, delta, dist)
    sol = integrate.newton_krylov(funct, x0, method='lgmres', verbose=1)
    S, I = np.split(sol, 2)
    S = np.concatenate(([S0], S))
    I = np.concatenate(([I0], I))
 
    times = np.linspace(0, num_days, num_days+1)
    factor = 1/delta # periods per day
    day_periods = [int(k*factor) for k in times ]
    S = np.array([S[k] for k in day_periods])
    I = np.array([I[k] for k in day_periods])
    R = N - S - I
    return pd.DataFrame(data={'Day':times, 'S': S, 'I':I,'R': R}).set_index('Day')

def test_sir_g2(delta=1):
    N = 1000
    dist = utils.stats.get_gamma(10, 2)
    model1 = stochasticSIR(population=N, disease_time_distribution = dist, delta = delta)
    model2 = stochasticSIR2(population=N, disease_time_distribution = dist, delta = delta)
    model1.plot()
    model2.plot()
    print_error(model1, model2, N)

if __name__ == '__main__':
    test_sir_g2()
