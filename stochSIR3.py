# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:50:30 2020

@author: germanr
"""


# The integral component
def deriv2(t, y, betaN, gam, times, S_guess, I_guess, survival, delta_loss, delta):
    # This function does not depend on y, but it depends on t
    # We buid piece-wise linear functions
    if t < min(times) or t > max(times):
        print(f"Error t={t}")
    S = interp1d(times, S_guess, fill_value='extrapolate')
    I = interp1d(times, I_guess, fill_value='extrapolate')
    dSdt = -betaN * S(t) * I(t)
    n = int(round(t /delta))
    dIdt = -dSdt - betaN * compute_integral(n, delta, S_guess, I_guess, survival, delta_loss)
    # print(f"S'({t})={dSdt}  I'({t})={dIdt}")
    return dSdt, dIdt


def stochasticSIR2(population=1000,
                   reproductive_factor=2.0,
                   disease_time_distribution=stats.expon(scale=10),
                   I0 = 1.0,
                   R0 = 0.0,
                   num_days = 160,
                   delta = 1):
    # Total population, N.
    N = population
    dist = disease_time_distribution
    # A grid of time points (in days)
    times = np.linspace(0, num_days, num_days + 1)
    
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gam, (in 1/days).
    gam = 1 / dist.mean()
    beta = reproductive_factor / dist.mean() 
    betaN = beta/N
    
    survival = dist.sf(times)
    loss_func = loss_function(dist)
    L = loss_func(times)
    delta_loss = np.zeros_like(L)
    delta_loss[:-1] = L[1:] - L[0:num_days]
    delta_loss[:0] = 0.0 # Last value
    
    # Initial conditions vector
    y0 = S0, I0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, times, args=(betaN, gam), tfirst=True)
    S_guess, I_guess = ret.T
    # Integrate again, with the integral, fixing guess 
    ret2 = odeint(func=deriv2, y0=y0, t=times, args=(betaN, gam, times, S_guess, I_guess, survival, delta_loss, delta), tfirst=True)
    S, I = ret2.T

    R = N - S - I
    return times, S, I, R


def residual(x, S0, I0, betaN, num_periods, delta, dist):
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
        integral[n] = simps(integrand, range(n+1))
    residual_S = dSdt - betaN * S * I 
    residual_I = dIdt - (betaN * S * I  - betaN * integral)
    return np.concatenate((residual_S, residual_I))

def stochasticSIR3(population=1000,
                   reproductive_factor=2.0,
                   disease_time_distribution=expon(scale=10),
                   I0 = 1.0,
                   R0 = 0.0,
                   num_days = 160,
                   delta = 1):
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
    sol = newton_krylov(funct, x0, method='lgmres', verbose=1)
    S, I = np.split(sol, 2)
    S = np.concatenate(([S0], S))
    I = np.concatenate(([I0], I))
 
    times = np.linspace(0, num_days, num_days+1)
    factor = 1/delta # periods per day
    day_periods = [int(k*factor) for k in times ]
    S = np.array([S[k] for k in day_periods])
    I = np.array([I[k] for k in day_periods])
    R = N - S - I
    return times, S, I, R