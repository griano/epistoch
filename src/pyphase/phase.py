# -*- coding: utf-8 -*-
r"""Module pyphase.phase.

This module allows you to work with Phase-Type distributions.

The CDF for PH variable is given by

.. math:: F(t) = 1 - \boldsymbol{\alpha} e^{\boldsymbol{A} t} \boldsymbol{1} \qquad x \geq 0,

where :math:`\boldsymbol{\alpha}` is a probability vector and the matrix :math:`\boldsymbol{A}` has
the transition rates between the phases.

To build a PH variable you need the vectoe ``alpha`` and matrix ``A``

.. doctest::

    >>> from pyphase import *
    >>> alpha = [0.1 , 0.9]
    >>> A = [[-2.0 , 1.0], [1.5, -3.0]]
    >>> v = phase(alpha, A)
    >>> print(v) #doctest: +REPORT_NDIFF
    PhaseType:
      alpha = [[0.1 0.9]]
      A     = [[-2.   1. ]
               [ 1.5 -3. ]]

After you have the variable you can do all operations that are typical with random variables in scipy.stats::

    >>> print(f"The mean is {v.mean():.3f}")
    The mean is 0.789
    >>> print(f"v.cdf(1.0)={v.cdf(1.0):.3f}")
    v.cdf(1.0)=0.721"""

import numpy as np
from numpy import matlib as ml
from scipy import linalg
from scipy.special import binom
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_frozen


def _solve_vector_A(vector, A):
    # Computes vector * A^(1), without actually inverting the matrix, by solving x * A = vector
    return np.linalg.solve(A.T, vector.T).T


class _PhaseTypeGen(rv_continuous):
    def _set_pars(self, alpha, A):
        alpha = ml.mat(alpha)
        A = ml.mat(A)
        self.alpha = alpha
        self.A = A
        self.alphaAi = None  # alpha * A^(-1), computed lazily
        self.erd = None  # Equilibrium residual distribution

    def _cdf1(self, x):
        return 1.0 - np.sum(self.alpha @ linalg.expm(self.A.A * x), axis=1) if x > 0 else 0.0

    def _cdf(self, x):
        return np.vectorize(self._cdf1)(x)

    def _pdf1(self, x):
        res = self.alpha @ linalg.expm(self.A.A * x) @ self._get_a() if x >= 0 else 0.0
        return res

    def _pdf(self, x):
        return np.vectorize(self._pdf1)(x)

    def _get_a(self):
        return -np.sum(self.A, axis=1)

    def _get_alpha_Ai(self):
        # Computes vector * A^(1), without actually inverting the matrix, by solving x * A = vector
        if self.alphaAi is None:
            self.alphaAi = _solve_vector_A(self.alpha, self.A)
        return self.alphaAi

    def _get_erd(self):
        if self.erd is None:
            gamma = 1 / self.mean()
            self.erd = phase(-gamma * self._get_alpha_Ai(), self.A)
        return self.erd

    def _moments(self, k=None):
        if k is None:
            k = 2
        moms = list()
        left = self.alpha
        for i in range(1, k + 1):
            left = -i * _solve_vector_A(left, self.A)
            moms.append(np.sum(left.flatten()))
        return moms

    def _stats(self):
        # Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
        moments = self._moments(4)
        moments = np.concatenate(([1.0], moments))
        mean = moments[1]
        centered_moments = {
            n: sum([binom(n, k) * moments[k] * (-mean) ** (n - k) for k in range(0, n + 1)]) for n in range(1, 5)
        }
        var = centered_moments[2]
        sd = np.sqrt(var)
        skew = centered_moments[3] / sd ** 3
        # kurtosis is fourth central moment / variance**2 - 3
        kurt = centered_moments[4] / (var * var) - 3.0
        return mean, var, skew, kurt


class PhaseType(rv_frozen):
    """
    Represent a Phase-Type distribution.

    Users should not call it directly but rather call ``phase(alpha, A)``
    """

    def __init__(self, alpha, A, dist, *args, **kwds):
        super(PhaseType, self).__init__(dist, *args, **kwds)
        self.alpha = ml.mat(alpha)
        self.A = ml.mat(A)
        self.dist._set_pars(alpha, A)

    def _pre_append(self, stg, pre):
        lines = stg.splitlines()
        result = [lines[0]]
        for line in lines[1:]:
            result.append(pre + line)
        return "\n".join(result)

    def __str__(self):
        options = {"precision": 2}
        alpha_stg = np.array2string(self.alpha, **options)
        A_stg = np.array2string(self.A, **options)
        return "PhaseType:\n" + f"  alpha = {alpha_stg}\n" + f"  A     = {self._pre_append(A_stg,'          ')}"

    def __repr__(self):
        return f"PH({self.alpha.__repr__()}, {self.A.__repr__()})"

    def params(self):
        r"""
        Returns the parameters of this PH distribution.

        Returns
        -------
        tuple
            (:math:`\pmb\alpha, \pmb A, \pmb a=- \pmb A \pmb 1`, dimension)

        """
        return self.alpha, self.A, self.dist._get_a(), self.A.shape[0]

    def loss1(self, x):
        r"""First order loss function.

        For a random variable C teh first order loss function is defined as

        .. math::  L(x) = [E(X-x)^+]

        Parameters
        ----------
        x: float
            The value at which the first order distribution is evaluated

        Returns
        -------
        float
            Value of the loss function at x.
        """
        return self.mean() * self.dist._get_erd().sf(x)

    def equilibrium_pi(self):
        r"""
        Return the equilibrium distribution of the associated PH distribution.

        In other word, it finds the vector \pi that solves

        .. math ::

            \pmb \pi = \pmb\pi(\pmb A + \pmb \alpha \pmb a), \qquad \pmb\pi\pmb 1=1

        Returns
        -------
        array
            vector :math"`\pmb\pi` that solves the equilibrium equations.
        """
        n = self.A.shape[0]
        a = self.dist._get_a()
        matrix = self.A + a @ self.alpha + np.ones((n, n))
        return np.linalg.solve(matrix.T, np.ones(n).T).T


def phase(alpha, A):
    r"""
    Creates a new PH variable.

    This method builds a PhaseType object by given the array :math:`\pmb\alpha` and matrix :math:`\pmb A`.
    The CDF for PH variable is given

    .. math:: F(t) = 1 - \boldsymbol{\alpha} e^{\boldsymbol{A} t} \boldsymbol{1} \qquad x \geq 0

    Parameters
    ----------
    alpha: array of float
        The initial probabilities vector. It must satisfy :math:`\sum_i \alpha_i \leq 1`.

    A: matrix of float
        The transition matrix between phases.

    Returns
    -------
    PH
        An object representing this distribution.
    """
    generator = _PhaseTypeGen(name="phase")
    frozen = generator.__call__()

    return PhaseType(alpha, A, frozen.dist)


def ph_expon(lambd=None, mean=None):
    """
    Builds an exponential distribution represented as PH.

    If mean is provided then lambda = 1/mean.

    Parameters
    ----------
    lambd : float, optional
        Rate value. One between lambd an mean must be given.
    mean : float, optional
        Mean value. One between lambd an mean must be given.

    Returns
    -------
    PH
        PH object.

    """
    return ph_erlang(1, lambd, mean)


def ph_erlang(n, lambd=None, mean=None):
    """
    Builds an Erlang(n,lmbda).

    If mean is provided then lambda = n/mean.

    Parameters
    ----------
    n : int
        The order of this Erlang
    lambd : float, optional
        Provide only one between lambd or mean
    mean : float, optional
        Provide only one between lambd or mean

    Returns
    -------
    PH
        PH representation for an Erlang

    """
    if lambd is None and mean is None:
        raise ValueError('"You must provide one between mean and lambda')
    lambd = lambd if lambd is not None else n / mean
    A = -lambd * np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = lambd
    alpha = np.zeros(n)
    alpha[0] = 1
    return phase(alpha, A)


def _z(n1, n2):
    # quiclky generates zeros
    return np.zeros((n1, n2))


def ph_sum(ph1, ph2):
    """
    Produces a new PH that is the sum of the given PH variables.

    Parameters
    ----------
    ph1: PH
        The first variable.
    ph2: PH
        The second variable.

    Returns
    -------
    PH
        The resulting PH variable.
    """
    alpha1, A1, a1, n1 = ph1.params()
    alpha2, A2, a2, n2 = ph2.params()
    alpha = np.block([alpha1, _z(1, n2)])
    A = np.block([[A1, a1 * alpha2], [_z(n2, n1), A2]])
    return phase(alpha, A)


def ph_mix(ph1, ph2, p1):
    """
    Produces a PH variable thet is a mixture of the two given PH variables.

    Parameters
    ----------
    ph1: PH
        The first variable.
    ph2: PH
        The second variable.
    p1: float
        Probability of choosing the first variable. 0 <= p1 <= 1

    Returns
    -------
    PH
        The resulting PH variable.
    """
    alpha1, A1, a1, n1 = ph1.params()
    alpha2, A2, a2, n2 = ph2.params()
    alpha = np.block([p1 * alpha1, (1 - p1) * alpha2])
    A = np.block([[A1, _z(n1, n2)], [_z(n2, n1), A2]])
    return phase(alpha, A)
