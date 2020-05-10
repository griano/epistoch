import numpy as np
from scipy import integrate, interpolate, optimize


def compute_integral(n, delta, S, I, times, survival, pdfs, loss1, dist, method="loss"):
    r"""

    Compute the integral needed for the integro-differential model.

    In other words, computes

    .. math:: \int_0^t g(t-x) I(x)S(x) dx \quad  \text{ for } t = n*\delta

    Parameters
    ----------
    n : integer
        upper limit for integral.
    delta : float
        interval size
    S : array
        Susceptible
    I : array
        Infected
    times: array
        times at which the arraya are evaluated
    survival : array of float
        array :math:`G_k \equiv P\{ T > k\delta\}`.
    pdfs: array of float
        array :math:`g(k\delta)`.
    loss1 array float
        :math:`L_k\equiv L(k\delta)`
    dist: rv_continuous
        Object that represents the distribution
    method: string
        One from "loss", "simpson" or "interpolate"
    Returns
    -------
    Value of the integral

    """
    if n == 0:
        return 0.0
    if method == "loss":
        IS = np.zeros_like(survival)
        # next line equivalent to: IS[: n + 1] = np.array([I[n - k] * S[n - k] for k in range(n + 1)])
        IS[: n + 1] = np.flipud(I[: n + 1] * S[: n + 1])
        slopes = np.diff(IS, append=0.0) / delta  # m1, m2, ...
        delta_slopes = np.diff(slopes, prepend=0.0)
        return IS[0] + sum(delta_slopes * loss1)

    if method == "simpson":
        integral_points = [pdfs[n - k] * S[k] * I[k] for k in range(0, n + 1)]
        return integrate.simps(integral_points, dx=delta)
    if method == "interpolate":
        t = n * delta
        interpolator = interpolate.interp1d(times[: n + 1], S[: n + 1] * I[: n + 1])

        def _integrand(tau):
            return dist.pdf(t - tau) * interpolator(tau)

        return integrate.quad(_integrand, 0, t)[0]


def get_total_infected(reproductive_factor, normalized_s0=1):
    r"""
    Estimate the total number of infected for a given reproductive factor.

    Find the value z that solves

    .. math:: 1-z = S_0 * e^{{\mathcal R_0} z},

    where :math:`\mathcal R_0` is the ``reproductive_factor``, and :math:`S_0` is the (normalized) initial
    population ``normalized_s0``.

    Parameters
    ----------
    reproductive_factor : float
        Basic reproductive factor.
    normalized_s0 : float, optional
        Initial fraction of population that is infected. The default is 1.

    Returns
    -------
    float
        The stimated fraction of total infected.

    """
    if reproductive_factor < 1:
        return 0.0
    else:
        fun = lambda x: 1 - x - normalized_s0 * np.exp(-reproductive_factor * x)
        result = optimize.root_scalar(fun, bracket=[0.0001, 1])
        return result.root


def _compute_array_error(name, x1, x2, N, do_print=True):
    # Convenience method to compute the error difference in tow array
    error = np.abs(x1 - x2) / N
    if do_print:
        print(f"{name}: max error = {np.max(error):.2}, avg error = {np.mean(error):.2}")
    return np.max(error)


def _compute_error(model1, model2, N, do_print=True):
    # Convenience methods to compute errors between two models
    if model1["population"] != model2["population"]:
        raise ValueError("We can only compare models with the same population")
    N = model1["population"]
    error_i = _compute_array_error("I", model1["data"].I, model2["data"].I, N, do_print)
    error_s = _compute_array_error("S", model1["data"].S, model2["data"].S, N, do_print)
    return 0.5 * (error_i + error_s)


def print_error(model1, model2):
    """
    Print error difference between two models

    Parameters
    ----------
    model1
        Result of a SIR, SIR_G model
    model2
        Result of a SIR, SIR_G model

    Returns
    -------

    """
    return _compute_error(model1, model2, True)


def report_summary(result):
    """
    Report a summary for a model.

    Parameters
    ----------
    result:
        Result from SIR like model
    """
    model = result["data"]
    name = result["name"]
    N = result["population"]
    n = len(model) - 1
    print(f"Model {name} Summary")
    print(f"  Total Infected People: {int(model.R[n]):,d} ({model.R[n]/N:.2%})")
    print(f"  Infection Peak: {int(np.max(model.I)):,d} ({np.max(model.I)/N:.2%})")
    print(f"  Peak Day: {int(np.argmax(model.I)):,d}")
    print(f"  Theoretical Total Infected: {result['total_infected']:.2%}")
