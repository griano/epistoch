# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:50:11 2020

@author: Germán Riaño, PhD
"""

import numpy as np

from epistoch.sir_phg import sir_phg
from pyphase.phase import ph_erlang, ph_mix, ph_sum


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
    return result
