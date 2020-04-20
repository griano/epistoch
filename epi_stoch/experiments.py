# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:34:26 2020

@author: germanr
"""

import logging
import scipy
import numpy as np
import os

import matplotlib.pyplot as plt
from timeit import timeit
import pandas as pd

import epi_stoch.utils as utils
from epi_stoch.stochSIR import classicalSIR, stochasticSIR, plot_sir, print_error, plot_IR, report_summary



def performance_test(N):   
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
    

def compare_models(name, dist, N=1000000, I0=1, num_days=100, R0=2.25):
    name1 = name + ":SIR"
    timesc, Sc, Ic, Rc = classicalSIR(population=N,
                                      I0=I0,
                                      reproductive_factor=R0,
                                      infectious_period_mean=dist.mean(), 
                                      num_days=num_days)
    fig = plot_sir(timesc, Sc, Ic, Rc, N, name1)    
    
    name2 = name + ":SIR-G"
    times, S, I, R = stochasticSIR(population=N, I0=I0,
                                   num_days=num_days, 
                                   delta=num_days/2000,
                                   reproductive_factor=R0, 
                                   disease_time_distribution=dist,
                                   method='loss')
    plot_sir(times, S, I, R, N, name2, fig, linestyle='--')    
    plt.show()

    report_summary(name1, Sc, Ic, Rc, N)
    report_summary(name2, S, I, R, N)
    print_error(Ic, I, Sc, S, N)
    filename = os.path.join("../paper/epistoch/figures/", name + "SIR-comp.pdf")
    print(f"Saving picture in file {os.path.abspath(filename)}")
    fig.savefig(filename, bbox_inches='tight')

    fig2 = plot_IR(Sc, Rc, Ic, N, label=name1)
    plot_IR(S, R, I, N, name2, fig2, linestyle='--')
    
    plt.show()    
    # save as PDF
    filename = os.path.join("../paper/epistoch/figures/", name + "IR-comp.pdf")
    print(f"Saving picture in file {os.path.abspath(filename)}")
    fig2.savefig(filename, bbox_inches='tight')
    

def variance_analysis(N = 1000000, I0 = 1, num_days = 100, R0 = 2.25, infectious_period_mean = 4.4, cvs=[.5, 1., 2.]):

    for cv in cvs:
        infectious_period_sd = cv * infectious_period_mean
        dists = {}
        dists['gamma'] =  utils.stats.get_gamma(infectious_period_mean, infectious_period_sd)
        dists['lognorm'] = utils.stats.get_lognorm(infectious_period_mean, infectious_period_sd)
        for name, dist in dists.items():
            mod_name = f'{name}-{cv:.2}'
            logging.info('Running model comp ' + mod_name)
            num_days = round(100 * max(1, cv))
            compare_models(mod_name + '-', dist, num_days=num_days)



if __name__ == '__main__':
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.basicConfig()
    compare_models('DIVOC', scipy.stats.norm(loc=4.5, scale=1) )
    variance_analysis()
