========
EpiStoch
========


.. image:: https://img.shields.io/pypi/v/epistoch.svg
        :target: https://pypi.python.org/pypi/epistoch

.. image:: https://travis-ci.com/griano/epistoch.svg?branch=master
    :target: https://travis-ci.com/griano/epistoch

.. image:: https://readthedocs.org/projects/epistoch/badge/?version=latest
        :target: https://epistoch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Epidemics Models with Random Infectious Period
----------------------------------------------

This software allows you to model epidemics with general random distribution for the infectious period.

Traditional epidemiology models like SIr do not take into account the distribution for the length of
the infectious period. In this software we include three functions that compute this models

.. image:: ./_static/DIVOC-SIR-comp.png

In this graph you can see how different the predictions are for the regular SIR modle with respect to SIR-G that actually uses
a more realistic distribution for the infectious period.
In SIR-G case the peak of infection occurs before and has a bigger intensity.
The number of individuals that get infected, however, remains thesame for both models

Models
------
* SIR: Classical SIR model, with (implied) exponential infectious period.  :py:func:`epistoch.sir_g.sir_classical`
* SIR_G: Like the classical SIR model, but with an arbitrary distribution. :py:func:`epistoch.sir_g.sir_g`
* SIR-PHG: A SIR model with Phase-Type distributions for the infectious period. :py:func:`epistoch.sir_phg.sir_phg`
* SEIRD: A SEIRD Model with hase-Type distributions for each stage. :py:func:`epistoch.seird_ph.seird_ph`


Notes
-----
* The theoretical foundation of the method is explained in this paper_.
* Source Code: https://github.com/griano/epistoch.
* Free software: MIT license. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.




.. _paper: https://github.com/griano/epistoch/blob/master/paper/epistoch/epi_stoch.pdf
