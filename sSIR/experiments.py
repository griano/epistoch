# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:34:26 2020

@author: germanr
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from timeit import timeit

import sSIR.utils as utils
from sSIR.stochSIR import classicalSIR, stochasticSIR, plot_sir, print_error, plot_RI 



def performance_test():   
    tt=timeit(setup="from __main__ import classicalSIR",
              stmt="tc, Sc, Ic, Rc = classicalSIR(population=1000, use_odeint=True)",
              number=1000)
    print(f"tt odeint = {tt}")
    tt=timeit(setup="from __main__ import classicalSIR",
              stmt="tc, Sc, Ic, Rc = classicalSIR(population=1000, use_odeint=False)",
              number=1000)
    print(f"tt ivp = {tt}")
    tc, Sc, Ic, Rc = classicalSIR(population=N, use_odeint=True)
    tc2, Sc2, Ic2, Rc2 = classicalSIR(population=N, use_odeint=False)
    print(np.max(np.abs(Ic-Ic2))/N)
    print(np.max(np.abs(Sc-Sc2))/N)
    
    
    
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

N = 1000
I0 = 1 # N/1000
num_days = 60
infectious_period_mean = 4.4
cv = 2
infectious_period_sd = cv * infectious_period_mean
R0 = 2.25

dist = utils.stats.get_gamma(infectious_period_mean,infectious_period_sd)
dist = utils.stats.get_lognorm(infectious_period_mean,infectious_period_sd)


timesc, Sc, Ic, Rc = classicalSIR(population=N, I0=I0, reproductive_factor=R0, infectious_period_mean=infectious_period_mean, 
                                  num_days=num_days)
fig = plot_sir(timesc, Sc, Ic, Rc, N, "classic")    

times, S, I, R = stochasticSIR(population=N, I0=I0, num_days=num_days, delta=num_days/1000, reproductive_factor=R0, 
                               disease_time_distribution=dist, method='loss')
plot_sir(times, S, I, R, N, "stoc", fig, linestyle='--')    
plt.show()

print_error(Ic, I, Sc, S, N)

fig2 = plot_RI(Sc, Rc, Ic, N, label='class')
plot_RI(S, R, I, N, 'stoc', fig2, linestyle='--')

plt.show()