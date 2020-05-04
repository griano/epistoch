=====
Usage
=====

In this example we compute a SIR-G model with Gamma distribution::



    from epistoch import *
    from epistoch.utils.plotting import plot_sir
    from scipy import stats
    import matplotlib.pyplot as plt


    # Let's build a SIR-G model

    dist = stats.gamma(a=2, scale=10)
    # The expected time is 20 days

    SIR_general = sir_g(
        name="SIR-G-Example",
        population=1000,
        num_days=160,
        reproductive_factor=2.2,
        infectious_time_distribution=dist,
        method="loss",
    )

    # Report a summary
    report_summary(SIR_general)

    # Now plot the result
    plot_sir(SIR_general)
    plt.show()

    # The data can be seen in
    print(SIR_general["data"])
