"""Step 3 - Non-linear distortion detection (Python port of
``Examples/Step_3_NonlinearDistortions.m``).

Uses the random odd-odd multisine experiment (``MultisineTypeB.mat``) to split
the output spectrum into linear / even / odd / noise contributions.
"""

import numpy as np
import matplotlib.pyplot as plt

import fditools as fdi
from _data import load_typeB
from _plot import save_fig, show


def main():
    p, u, y = load_typeB()
    trans, trend = 1, 0
    x, _ = fdi.pretreat(u, p.nrofs, p.fs, trans, trend)
    y, _ = fdi.pretreat(y, p.nrofs, p.fs, trans, trend)

    Yl, freql, Yo, freqo, Ye, freqe, Yn, freqn = fdi.time2nld(
        x, y, p.fs, p.fl, p.fh, p.df)

    names = ["motor angle", "load angle"]
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for o in range(2):
        axs[o].semilogx(freql, fdi.dbm(Yl[:, o]), label="Y linear")
        axs[o].semilogx(freqe, fdi.dbm(Ye[:, o]), label="Y even")
        axs[o].semilogx(freqo, fdi.dbm(Yo[:, o]), label="Y odd")
        axs[o].semilogx(freqn, fdi.dbm(Yn[:, o]), label="Y noise")
        axs[o].set_ylabel(f"{names[o]}\nMagnitude [dB]")
        axs[o].set_xlim(p.fl, p.fh)
    axs[0].legend(fontsize=8, ncol=4)
    axs[1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Non-linear distortion analysis")
    save_fig(fig, "step3_nonlinear.png")
    show()


if __name__ == "__main__":
    main()
