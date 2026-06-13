"""Tutorial 3 (input non-linearity) - Python port of
``Examples/Tutorial_3_nonlinear_in.m``.

An odd-odd random multisine drives the plant; a static polynomial non-linearity
acts on the **input** (added to the output):

    y = G * u + odd_nl * u^3 + even_nl * u^2

``time2nld`` then separates the linear / even / odd / noise contributions at
several excitation amplitudes.
"""

import numpy as np
import control
import matplotlib.pyplot as plt

import fditools as fdi
from _data import benchmark_plant
from _plot import save_fig, show


def main():
    P0, label = benchmark_plant()
    print(f"true plant: {label}")

    harm = dict(fs=10000.0, df=1.0, fl=1.0, fh=1000.0, fr=1.02)
    options = dict(itp="r", ctp="c", dtp="O", gtp="l")     # odd-odd, linear grid
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    fs = harm["fs"]
    base = np.squeeze(ms.x[0, 0, :])

    nrofp = 20
    u0 = np.tile(base, nrofp)
    t = np.arange(u0.size) / fs
    odd_nl, even_nl = 1e-4, 1e-3
    out_noise = 1e-5 * np.random.default_rng(0).standard_normal(u0.size)

    cases = [("amp 1", 1.0), ("amp 0.01", 0.01), ("amp 100", 100.0),
             ("noise-free", 1.0)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, amp) in zip(axs.ravel(), cases):
        u = amp * u0
        y_lin = control.forced_response(P0, t, u).outputs
        if name == "noise-free":
            y = y_lin
        else:
            y = y_lin + odd_nl * u ** 3 + even_nl * u ** 2 + out_noise

        x, _ = fdi.pretreat(u, ms.nrofs, fs, 2, 0)
        yy, _ = fdi.pretreat(y, ms.nrofs, fs, 2, 0)
        Yl, fl_, Yo, fo, Ye, fe, Yn, fn = fdi.time2nld(x, yy, fs, harm["fl"], harm["fh"], harm["df"])
        ax.semilogx(fl_, fdi.dbm(Yl[:, 0]), "*", ms=3, label="Y lin")
        ax.semilogx(fe, fdi.dbm(Ye[:, 0]), "*", ms=3, label="Y even")
        ax.semilogx(fo, fdi.dbm(Yo[:, 0]), "*", ms=3, label="Y odd")
        ax.semilogx(fn, fdi.dbm(Yn[:, 0]), "*", ms=3, label="Y noise")
        ax.set_xlim(harm["fl"], harm["fh"])
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title(name)
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Input non-linearity ({label})")
    save_fig(fig, "tutorial_3_nonlinear_in.png")
    show()


if __name__ == "__main__":
    main()
