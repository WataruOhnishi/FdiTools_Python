"""Step 4 - Parametric estimation (Python port of
``Examples/Step_4_ParametricEstimation.m``).

SIMO (1 input, 2 outputs) estimation from the ``MultisineTypeA.mat`` motor-bench
data with deterministic (WLS/NLS) and stochastic (ML/BTLS) estimators.
"""

import numpy as np
import matplotlib.pyplot as plt

import fditools as fdi
from _data import load_typeA
from _plot import save_fig, show


def main():
    ms, u, y = load_typeA()
    x, _ = fdi.pretreat(u, ms.nrofs, ms.harm.fs, 1, 0)
    y, _ = fdi.pretreat(y, ms.nrofs, ms.harm.fs, 1, 0)
    Pest = fdi.time2frf_ml(x, y, ms)
    freq = Pest.freq

    n = 4
    mh = np.array([[2], [0]])      # numerator orders per output (SIMO)
    ml = np.array([[0], [0]])
    relvar, itr, GN, cORd, relax = 1e-10, 500, 0, "c", 1.0
    FRF_W = np.ones((freq.size, 2))

    SYS = {}
    SYS["nls"], SYS["wls"] = fdi.nlsfdi(Pest, FRF_W, n, mh, ml, itr, relvar, GN, cORd)
    SYS["ml"], SYS["ls"] = fdi.mlfdi(Pest, n, mh, ml, itr, relvar, GN, cORd)
    SYS["btls"], SYS["gtls"] = fdi.btlsfdi(Pest, n, mh, ml, relax, itr, relvar, cORd)

    FRF = {k: fdi.hfrf(v, freq) for k, v in SYS.items()}
    meas = Pest.frf_columns()                       # (nroff, 2)
    sG = Pest.userdata.sG
    FRFn = Pest.userdata.FRFn

    titles = ["Motor-side (H11)", "Load-side (H21)"]

    def phase_deg(resp, ref=None):
        """Unwrapped phase [deg]; if *ref* given, shifted by k*360 to share its
        branch (phase is only defined modulo 360 deg)."""
        ph = np.unwrap(np.angle(resp)) * 180.0 / np.pi
        if ref is not None:
            ph = ph + 360.0 * np.round(np.median(ref - ph) / 360.0)
        return ph

    def bode_panel(fig, est_keys, noise, noise_label, suptitle):
        axs = fig.subplots(2, 2)
        for o in range(2):
            axs[0, o].semilogx(freq, fdi.dbm(meas[:, o]), label="FRF")
            for key in est_keys:
                axs[0, o].semilogx(freq, fdi.dbm(FRF[key][:, o]), label=key.upper())
            axs[0, o].semilogx(freq, fdi.dbm(noise[:, o]), "m--", label=noise_label)
            axs[0, o].set_title(titles[o])
            axs[0, o].set_ylabel("Amplitude [dB]")
            axs[0, o].set_xlim(10, 300)
            axs[0, o].legend(fontsize=7)
            ref = phase_deg(meas[:, o])
            axs[1, o].semilogx(freq, ref)
            for key in est_keys:
                axs[1, o].semilogx(freq, phase_deg(FRF[key][:, o], ref))
            axs[1, o].set_ylabel("Phase [deg]")
            axs[1, o].set_xlabel("frequency [Hz]")
            axs[1, o].set_xlim(10, 300)
        fig.suptitle(suptitle)

    fig1 = plt.figure(figsize=(11, 7))
    bode_panel(fig1, ["wls", "nls", "ls"], FRFn, "FRFn", "Deterministic estimators")
    save_fig(fig1, "step4_deterministic.png")

    fig2 = plt.figure(figsize=(11, 7))
    bode_panel(fig2, ["ml", "btls", "gtls"], sG, "sG", "Stochastic estimators")
    save_fig(fig2, "step4_stochastic.png")

    show()
    return SYS


if __name__ == "__main__":
    main()
