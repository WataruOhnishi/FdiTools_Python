"""Step 2 - Non-parametric FRF (Python port of ``Examples/Step_2_NonparametricFRF.m``).

Loads the Schroeder-multisine motor-bench measurement (``MultisineTypeA.mat``),
pre-treats the periodic data and estimates the FRF (motor- and load-side) with
the maximum-likelihood method.
"""

import numpy as np
import matplotlib.pyplot as plt

import fditools as fdi
from _data import load_typeA
from _plot import save_fig, show


def main():
    ms, u, y = load_typeA()
    trans, trend = 1, 0
    x, time = fdi.pretreat(u, ms.nrofs, ms.harm.fs, trans, trend)
    y, time = fdi.pretreat(y, ms.nrofs, ms.harm.fs, trans, trend)
    nrofp = u.shape[0] // ms.nrofs - trans
    print(f"{nrofp} periods after {trans} transient removal")

    # raw periodic time data
    fig0, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for k in range(nrofp):
        sl = slice(k * ms.nrofs, (k + 1) * ms.nrofs)
        axs[0].plot(time[:ms.nrofs], x[sl])
        axs[1].plot(time[:ms.nrofs], y[sl, 0])
        axs[2].plot(time[:ms.nrofs], y[sl, 1])
    axs[0].set_ylabel("input current [A]")
    axs[1].set_ylabel("motor angle [rad]")
    axs[2].set_ylabel("load angle [rad]")
    axs[2].set_xlabel("time [s]")
    axs[0].set_title(f"{nrofp} periods ({trans} transient removed)")
    save_fig(fig0, "step2_timedata.png")

    # non-parametric ML estimation
    Pest = fdi.time2frf_ml(x, y, ms, flagTime=True)

    fig1, _ = fdi.bode_fdi(Pest[0, 0], noise=(Pest.freq, Pest.userdata.sGhat[:, 0]),
                           title="Motor-side FRF", labels=["FRF", "sGhat"])
    save_fig(fig1, "step2_frf_motor.png")
    fig2, _ = fdi.bode_fdi(Pest[1, 0], noise=(Pest.freq, Pest.userdata.sGhat[:, 1]),
                           title="Load-side FRF", labels=["FRF", "sGhat"])
    save_fig(fig2, "step2_frf_load.png")

    show()
    return Pest


if __name__ == "__main__":
    main()
