# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:14:05 2020

@author: Germán Riaño
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def set_plt_for_latex():
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "serif", "sans-serif": ["Palatino"]})
    plt.set_loglevel("info")


def prepare_plot(title, use_latex=False):

    if use_latex:
        set_plt_for_latex()

    fig = plt.figure(facecolor="w")
    # use LaTeX fonts in the plot

    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True) # Gray foreground
    ax = fig.add_subplot(111, axisbelow=True)
    ax.set_xlabel("Time /days")
    ax.set_ylabel("Population Percent")
    # ax.set_ylim(0,1.1)
    ax.grid(b=True, which="major", c="k", lw=0.5, ls=":")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    return fig, ax


def plot_sir(
    model,
    fig=None,
    title=None,
    formats={"S": "b-", "I": "r-", "R": "g-"},
    legend_fmt={"loc": "best", "shadow": True},
    use_latex=False,
    **kwd,
):
    data = model["data"]
    name = model["name"]
    N = model["population"]
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    if fig is None:
        title = name if title is None else title
        fig, ax = prepare_plot(title, use_latex)
    else:
        allaxes = fig.get_axes()
        ax = allaxes[0]
    t = data.index
    for s, fmt in formats.items():
        series = data[s] / N
        linewidth = kwd.get("linewidth", 2)
        linestyle = kwd.get("linestyle", "-")
        ax.plot(
            t, series, fmt, alpha=0.8, lw=linewidth, linestyle=linestyle, label=f"{s} - {name}",
        )
    ax.legend(**legend_fmt)
    return fig


def plot_IR(model, fig=None, linestyle="-", title=None, legend_fmt={"loc": "best", "shadow": True}, use_latex=False):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    data = model["data"]
    name = model["name"]
    N = model["population"]
    if fig is None:
        title = name if title is None else title
        fig, ax = prepare_plot(title, use_latex)
    else:
        allaxes = fig.get_axes()
        ax = allaxes[0]
    ax.plot(
        data.S / N, data.R / N, "g", alpha=0.8, lw=2, linestyle=linestyle, label="Removed - " + name,
    )
    ax.plot(
        data.S / N, data.I / N, "b", alpha=0.8, lw=2, linestyle=linestyle, label="Infected - " + name,
    )
    ax.legend(**legend_fmt)
    return fig
