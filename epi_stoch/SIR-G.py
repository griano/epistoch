# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:20:32 2020

@author: German Riano (griano@jmarkov.org)
"""


import logging
import math

import numpy as np
import pandas as pd
from scipy import integrate, stats, interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm

from epi_stoch.utils.stats import loss_function

EPS = 1e-5

# The SIR model differential equations.
def deriv(t, y, beta, gam):
    S, I = y
    dSdt = -beta * S * I
    dIdt =  beta * S * I - gam * I
    return dSdt, dIdt

def classicalSIR(population=1000,
                 reproductive_factor=2.0,
                 infectious_period_mean=10,
                 I0 = 1.0,
                 R0 = 0.0,
                 num_days = 160,
                 use_odeint=True):
    """Code based on
    https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    """
    # Total population, N.
    N = population
    # Initial number of infected and recovered individuals, I0 and R0.
    I0 = I0/N
    R0 = R0/N

    # Everyone else, S0, is susceptible to infection initially.
    S0 = 1 - I0 - R0
    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1./infectious_period_mean 
    beta = reproductive_factor * gam

    # A grid of time points (in days)
    times = np.linspace(0, num_days, num_days+1)
    
   # Initial conditions vector
    y0 = [S0, I0]
    # Integrate the SIR equations over the time grid, t.
    if use_odeint:
        ret = integrate.odeint(deriv, y0, times, args=(beta, gam), tfirst=True)
        S, I = ret.T
    else:
        sol = integrate.solve_ivp(fun=deriv, 
                        t_span=[0,num_days],
                        y0=y0, 
                        args=(beta, gam),
                        t_eval=times,
                        vectorized=True)
        times = sol.t
        S, I  = sol.y
    S = S * N
    I = I * N        
    R = N - S - I
    return times, S, I, R

def compute_integral(n, delta, S, I, times, survival, pdfs, loss, dist, method="loss"):
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
    if n == 0 : 
        return 0.0
    if method == 'loss':
        IS = np.zeros_like(survival)
        IS[:n+1] = np.array([ I[n-k]*S[n-k] for k in range(n+1) ])
        slopes = np.diff(IS, append=0.0)/delta  # m1, m2, ...
        delta_slopes = np.diff(slopes, prepend=0.0)
        return IS[0] + sum(delta_slopes*loss)
    if method == 'simpson':
        integral_points = [ pdfs[n-k]*S[k]*I[k] for k in range(0, n+1) ]
        return integrate.simps(integral_points, dx=delta)
    if method == 'interpolate':
        t = n * delta
        interpolator = interpolate.interp1d(times[:n+1], S[:n+1]*I[:n+1])
        integrand = lambda tau: dist.pdf(t-tau) * interpolator(tau)
        return integrate.quad(integrand, 0, t)[0]




def stochasticSIR(population=1000, 
                  reproductive_factor=2.0,
                  disease_time_distribution=stats.expon(scale=10), 
                  I0 = 1.0, 
                  R0 = 0.0,
                  num_days = 160,
                  delta = 1,
                  method='loss'):
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
    logging.info(f"Computing SIR-G model with {num_periods} periods.")
    for n in tqdm(range(0, num_periods)):
        S[n+1] = S[n] - delta * beta * S[n] * I[n]
        integral = compute_integral(n, delta, S, I, times, survival, pdfs, loss1, dist, method=method)
        # Expected case for exponential variables    
        integral_expon = (gam/beta) * (I[n] - I0 * survival[n])
        logging.debug(f"n={n}, t={n*delta}, integral = {integral}, integral_expon = {integral_expon}")
        I[n+1] = I[n]  + delta * ( beta * (S[n] * I[n] - integral ) - gam * I0 * survival[n]) 
        # I[n+1] = I[n]  + delta * (beta * S[n]  - gam ) * I[n] 

    # Report one point per day, and de-normalize
    days = np.linspace(0, num_days, num_days + 1)
    S = N * interpolate.interp1d(times,S)(days)
    I = N * interpolate.interp1d(times,I)(days)
    R = N - S - I
    return days, S, I, R



def plot_sir(t, S, I, R, N, label, fig=None, linestyle='-'):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if fig is None:
        fig = plt.figure(facecolor='w')
        # use LaTeX fonts in the plot
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family':'serif', 'sans-serif':['Palatino']})
         
        # ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Population Percent')
        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
    else:
        allaxes = fig.get_axes()
        ax = allaxes[0]    
    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, linestyle=linestyle, label='Susceptible-' + label)
    ax.plot(t, I/N, 'r', alpha=0.5, lw=2, linestyle=linestyle, label='Infected-' + label)
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, linestyle=linestyle, label='Removed-' + label)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    return fig
 
def plot_IR(S, R, I, N, label, fig=None, linestyle='-'):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if fig is None:
        fig = plt.figure(facecolor='w')
        # ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
        ax = fig.add_subplot(111, axisbelow=True)
        ax.set_xlabel('Susceptibe')
        ax.set_ylabel('Removed/Infected')
        ax.set_ylim(0,1.2)
        ax.set_xlim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
    else:
        allaxes = fig.get_axes()
        ax = allaxes[0]    
    ax.plot(S/N, R/N, 'g', alpha=0.5, lw=2, linestyle=linestyle, label='Removed-' + label)
    ax.plot(S/N, I/N, 'b', alpha=0.5, lw=2, linestyle=linestyle, label='Infected-' + label)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    return fig


def print_array_error(x1, x2, N, name):
    error = np.abs(x1-x2)/N
    print(f"{name}: max error = {np.max(error):.2}, avg error = {np.mean(error):.2}")

def print_error(I1, I2, S1, S2, N):
    print_array_error(I1, I2, N, 'I')
    print_array_error(S1, S2, N, 'S')

def report_summary(name, S, I, R, N):
    n = len(R) - 1
    print(f"Model {name} Summary")
    print(f"  Total Infected People: {int(R[n]):,d} ({R[n]/N:.2%})")
    print(f"  Infection Peak: {int(np.max(I)):,d} ({np.max(I)/N:.2%})")
    print(f"  Peak Day: {int(np.argmax(I)):,d}")
    

