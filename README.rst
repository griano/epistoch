========
EpiStoch
========


.. image:: https://img.shields.io/pypi/v/epistoch.svg
        :target: https://pypi.python.org/pypi/epistoch

.. image:: https://img.shields.io/travis/griano/epistoch.svg
        :target: https://travis-ci.com/griano/epistoch

.. image:: https://readthedocs.org/projects/epistoch/badge/?version=latest
        :target: https://epistoch.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Epidemics Models with Random Infectious Period
----------------------------------------------

This software allows you to model epidemics with general random distribution for the infectious period.

Traditional epidemiology models, like SIR, do not take into account the distribution for the length of
the infectious period. In this software, we include three functions that compute these type of models
using other distributions.

.. image:: https://epistoch.readthedocs.io/en/latest/_images/DIVOC-SIR-comp.png

In this graph you can see how different the predictions are for the regular SIR model with respect to SIR-G
that actually uses a more realistic distribution for the infectious period.
In SIR-G case the peak of infection occurs before, and has a bigger intensity.
The number of individuals that eventually get infected, however, remains the same for both models

Models
------
* SIR_G: Like the classical SIR model, but with an arbitrary distribution.
* SIR-PH: A SIR model with Phase-Type distributions for the infectious period
* SEIRD: A SEIRD Model with hase-Type distributions for each stage.

Notes
-----

* Paper_
* Documentation: https://epistoch.readthedocs.io.
* Free software: MIT license. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.




.. _Paper: https://github.com/griano/epistoch/blob/master/paper/epistoch/epi_stoch.pdf
