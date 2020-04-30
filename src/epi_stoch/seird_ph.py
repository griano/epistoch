# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:50:11 2020

@author: Germán Riaño, PhD
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from epi_stoch.sir_g import report_summary
from epi_stoch.sir_ph import sir_phg
from epi_stoch.utils.plotting import plot_IR, plot_sir
from pyphase.phase import ph_erlang, ph_expon, ph_mix, ph_sum


def seird_ph(
    population=1000,
    beta=0.2,
    exposed_time=ph_erlang(n=3, mean=14),
    die_time=ph_erlang(n=2, mean=7),
    recover_time=ph_erlang(n=4, mean=12),
    fatality_rate=0.2,
    I0=1,
    num_days=250,
    logger=None,
):

    second_stage = ph_mix(die_time, recover_time, fatality_rate)
    total_time = ph_sum(exposed_time, second_stage)
    print(total_time)

    _, _, ae, ne = exposed_time.params()
    _, _, ad, nd = die_time.params()
    _, _, ar, nr = recover_time.params()

    beta = beta * np.concatenate([np.zeros(ne), np.ones(nd), np.ones(nr)])
    print(beta)

    result = sir_phg(
        population=population,
        beta=beta,
        disease_time_distribution=total_time,
        I0=I0,
        R0=0.0,
        num_days=num_days,
        logger=logger,
        report_phases=True,
    )
    
    exposed_cols, diying_cols, recovering_cols = np.split(result['I-columns'], [ne, ne+nd] )
    data = result['data']
    data['E'] = data[exposed_cols].sum(axis=1)
    
    _, deaths_cols, recovered_cols = np.split(result['R-columns'], [ne, ne+nd] )
    data['D'] = data[deaths_cols].sum(axis=1)
    data['Rc'] = data[recovered_cols].sum(axis=1)
    data['Id'] = data['I'] # All infected (including E)
    data['I'] = data['I'] - data['E'] # Actual infectious
    data = data.drop(columns=result['I-columns'] + result['R-columns'])
    result['data'] = data
    return result


def test_seird(do_plot=False):
    N=1000
    beta=0.2

    exposed_time_mean = 1
    die_time_mean     = 7
    recover_time_mean = 12
    
    exposed_time_exp = ph_expon(mean=exposed_time_mean)
    die_time_exp     = ph_expon(mean=die_time_mean)
    recover_time_exp = ph_expon(mean=recover_time_mean)

    exposed_time = ph_erlang(n=10, mean=exposed_time_mean)
    die_time     = ph_erlang(n=15, mean=die_time_mean)
    recover_time = ph_erlang(n=8, mean=recover_time_mean)

    fatality_rate=0.2
    I0=10
    num_days=200
    
    model_expo = seird_ph(population=N,
                     beta=beta,
                     exposed_time=exposed_time_exp, 
                     die_time=die_time_exp,
                     recover_time=recover_time_exp,
                     fatality_rate=fatality_rate,
                     I0=I0,
                     num_days=num_days)
    model_gen = seird_ph(population=N,
                     beta=beta,
                     exposed_time=exposed_time, 
                     die_time=die_time,
                     recover_time=recover_time,
                     fatality_rate=fatality_rate,
                     I0=I0,
                     num_days=num_days)
    if do_plot:
        formats={"S": "b-", "E":"c-", "I": "r-", "R": "g-", "Rc": "y-", "D":"m-"}
        legend_fmt={"loc": "upper right", "shadow":True, "framealpha": 1.0, "bbox_to_anchor": (1, 1)}
        report_summary("SEIRD-Expo", model_expo, N)
        fig = plot_sir("SEIRD-Expo", model_expo, N, title="SEIRD Expo vs Erlang", formats=formats)
    
        report_summary("SEIRD-Gen", model_gen, N)
        fig = plot_sir("SEIRD-Erlang", model_gen, N, formats=formats, 
                       fig=fig, linestyle=":", legend_fmt=legend_fmt)
    
        plt.show()
        filename = os.path.join("./paper/epistoch/figures/", "SEIRD.pdf")
        print(f"Saving picture in file {os.path.abspath(filename)}")
        fig.savefig(filename, bbox_inches="tight")


if __name__ == '__main__':
    test_seird(do_plot=True)