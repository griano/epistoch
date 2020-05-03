# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:50:11 2020

@author: Germán Riaño, PhD
"""

import numpy as np

from epistoch.sir_phg import sir_phg
from pyphase.phase import ph_erlang, ph_mix, ph_sum


def seird_ph(
    name, population, beta, exposed_time, time_to_die, time_to_recover, fatality_rate, num_days, I0=1, logger=None,
):
    """
    Compute a SEIRD model with Phase-Type distribution for the different stages.

    Parameters
    ----------
    name : string
        Model name.
    population : int or float
        Population size.
    beta : float
        Contagion rate.
    exposed_time : PH distribution
        Distribution of time exposed, not yet infectious.
    time_to_die : PH distribution
        Time to die after becoming infectious.
    time_to_recover : PH distribution
        Time to recover after becoming infectious.
    fatality_rate : float
        Percentage of individuals that die.
    num_days : int
        Number of days ot analyze.
    I0 : int, optional
        Initial infected population. The default is 1.
    logger : Logger object, optional
        Logger object. If not given default logging is used.

    Returns
    -------
    result : dict
        Dictionary with fields:
            - name: model name
            - population: Total population
            - data: data Frame with columns
                - S : Susceptible,
                - E : Exposed,
                - I : Infectious (Dying or recovering),
                - Rc : Total Recovered,
                - D : Total deaths
                - R : Removed (R+D),
    """

    second_stage = ph_mix(time_to_die, time_to_recover, fatality_rate)
    total_time = ph_sum(exposed_time, second_stage)
    print(total_time)

    _, _, ae, ne = exposed_time.params()
    _, _, ad, nd = time_to_die.params()
    _, _, ar, nr = time_to_recover.params()

    beta = beta * np.concatenate([np.zeros(ne), np.ones(nd), np.ones(nr)])
    print(beta)

    result = sir_phg(
        name,
        population=population,
        beta=beta,
        infectious_time_distribution=total_time,
        I0=I0,
        num_days=num_days,
        logger=logger,
        report_phases=True,
    )

    exposed_cols, diying_cols, recovering_cols = np.split(result["I-columns"], [ne, ne + nd])
    data = result["data"]
    data["E"] = data[exposed_cols].sum(axis=1)

    _, deaths_cols, recovered_cols = np.split(result["R-columns"], [ne, ne + nd])
    data["D"] = data[deaths_cols].sum(axis=1)
    data["Rc"] = data[recovered_cols].sum(axis=1)
    data["Id"] = data["I"]  # All infected (including E)
    data["I"] = data["I"] - data["E"]  # Actual infectious
    data = data.drop(columns=result["I-columns"] + result["R-columns"])
    result["data"] = data
    result["Name"] = name
    result["population"] = population

    return result
