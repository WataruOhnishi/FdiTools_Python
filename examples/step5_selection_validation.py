"""Step 5 - Selection & validation (Python port of
``Examples/Step_5_SelectionValidation.m``).

Estimates the SIMO model with the classical (positional) calling convention and
runs the three validation tests: residual whiteness, cost function and
chi-squared.
"""

import numpy as np
import matplotlib.pyplot as plt

import fditools as fdi
from _data import load_typeA
from _plot import save_fig, show


def main():
    ms, u, y = load_typeA()
    fs, df, fl, fh = ms.harm.fs, ms.harm.df, ms.harm.fl, ms.harm.fh
    x, _ = fdi.pretreat(u, ms.nrofs, fs, 1, 0)
    y, _ = fdi.pretreat(y, ms.nrofs, fs, 1, 0)

    # non-parametric FRF (classical positional convention)
    X, Y, FRFs, FRFn, freq, sX2, sY2, cXY, sCR = fdi.time2frf_ml(
        x, y, fs=fs, fl=fl, fh=fh, df=df)

    n = 4
    mh = np.array([[2], [0]])
    ml = np.array([[0], [0]])
    relvar, itr, GN, cORd, relax = 1e-10, 500, 0, "c", 1.0
    FRF_W = np.ones_like(FRFs, dtype=float)

    SYS = {}
    SYS["nls"], SYS["wls"] = fdi.nlsfdi(FRFs, freq, FRF_W, n, mh, ml,
                                        itr, relvar, GN, cORd, fs)
    SYS["ml"], SYS["ls"] = fdi.mlfdi(X, Y, freq, sX2, sY2, cXY, n, mh, ml,
                                     itr, relvar, GN, cORd, fs)
    SYS["btls"], SYS["gtls"] = fdi.btlsfdi(X, Y, freq, n, mh, ml, sY2, sX2, cXY,
                                           relax, itr, relvar, cORd, fs)

    nrofp = x.shape[0] / fs * df

    # ----- Test 1: residual whiteness ----------------------------------
    lags, corr, cb50, frac50, tag, cb95, frac95 = fdi.residtest(
        x, y, freq, FRFs, SYS, sCR, fs)
    fig1, axs = plt.subplots(1, 2, figsize=(11, 4))
    for o in range(2):
        for m in range(corr.shape[1]):
            axs[o].plot(lags, corr[:, m, o], ".", ms=2, label=tag[o][m])
        axs[o].plot(lags, cb95, "k", lw=0.8, label="cb95")
        axs[o].plot(lags, cb50, "k--", lw=0.8, label="cb50")
        axs[o].set_title(f"output {o + 1} residual whiteness")
        axs[o].set_xlabel("lag number")
        axs[o].set_ylim(0, 1.5)
        axs[o].legend(fontsize=7)
    save_fig(fig1, "step5_residuals.png")

    # ----- Test 2: cost function ---------------------------------------
    cost, intv, tagc = fdi.costtest(X, Y, freq, sX2, sY2, cXY, SYS, relax, nrofp)
    fig2, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    xpos = np.arange(len(tagc))
    for h in range(cost.shape[1]):
        ax.bar(xpos + (h - 0.5) * width, cost[:, h], width, label=f"H{h + 1}1")
    ax.axhline(intv[1], color="k", ls="--", label="noise bound")
    ax.set_xticks(xpos)
    ax.set_xticklabels(tagc)
    ax.set_ylim(0, intv[1] * 5)
    ax.set_title("Estimator selection: residual cost")
    ax.legend(fontsize=8)
    save_fig(fig2, "step5_cost.png")

    # ----- Test 3: chi-squared -----------------------------------------
    err, var, tagx = fdi.chi2test(X, Y, freq, FRFs, sCR, SYS)
    fig3, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for o in range(2):
        for m in range(err.shape[1]):
            axs[o].loglog(freq, err[:, m, o], lw=0.8, label=tagx[o][m])
        axs[o].loglog(freq, var[:, o], "k", lw=1.2, label="CRLB")
        axs[o].set_ylabel(f"output {o + 1} error^2")
        axs[o].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel("frequency [Hz]")
    fig3.suptitle("Chi-squared modelling error")
    save_fig(fig3, "step5_chi2.png")

    show()


if __name__ == "__main__":
    main()
