"""Tutorial 2 (iterative) - Python port of ``Examples/Tutorial_2_iterative.m``.

Combines three multisine experiments (wideband qlog, an inverse-S/N linear band
and a high-frequency qlog band) into a single FRF with ``fcat_fdi``/``fdel_fdi``,
then fits a parametric model.
"""

import numpy as np
import control
import matplotlib.pyplot as plt

import fditools as fdi
from _data import benchmark_plant
from _plot import save_fig, show


def run_experiment(P0, harm, options, Hampl, nrofp, fs,
                   inputnoize=0.01, outputnoize=0.003, trans=2, seed=0):
    """Design a multisine, run a noisy periodic experiment, estimate the FRF."""
    ms = fdi.multisine(harm, Hampl, options)
    rng = np.random.default_rng(seed)
    u = np.tile(np.squeeze(ms.x[0, 0, :]), nrofp)
    u = u + inputnoize * rng.standard_normal(u.size)
    t = np.arange(u.size) / fs
    y = control.forced_response(P0, t, u).outputs
    y = y + outputnoize * rng.standard_normal(y.shape)
    x, _ = fdi.pretreat(u, ms.nrofs, fs, trans, 0)
    y, _ = fdi.pretreat(y, ms.nrofs, fs, trans, 0)
    return fdi.time2frf_ml(x, y, ms)


def main():
    P0, label = benchmark_plant()
    print(f"true plant: {label}")
    fs = 10000.0

    # Exp 1: wideband quasi-log excitation
    h1 = dict(fs=fs, df=1.0, fl=1.0, fh=1000.0, fr=1.02)
    opt_q = dict(itp="r", ctp="c", dtp="f", gtp="q")
    Pest_q = run_experiment(P0, h1, opt_q, control.tf([1], [1]), 5, fs, seed=1)
    Pest_q_del = fdi.fdel_fdi(Pest_q, 100.0, 1000.0)     # keep 1..100 Hz

    # Exp 2: linear band 101..296 Hz weighted by inverse S/N from Exp 1
    sG_inv = np.abs(Pest_q.userdata.sG[:, 0]) / np.abs(Pest_q.response[0, 0, :])
    Hampl = fdi.FrfData(sG_inv, Pest_q.freq)
    h2 = dict(fs=fs, df=1.0, fl=101.0, fh=296.0, fr=1.02)
    Pest_lin = run_experiment(P0, h2, opt_q, Hampl, 20, fs, seed=2)

    # Exp 3: high-frequency quasi-log band 302..1000 Hz
    h3 = dict(fs=fs, df=1.0, fl=302.0, fh=1000.0, fr=1.02)
    Pest_hi = run_experiment(P0, h3, opt_q, control.tf([1], [1]), 20, fs, seed=3)

    # Combine the three experiments (lower-noise line wins on overlap)
    Pest = fdi.fcat_fdi(Pest_q_del, Pest_lin, Pest_hi)
    print(f"combined FRF: {Pest.freq.size} lines "
          f"({Pest.freq[0]:.0f}..{Pest.freq[-1]:.0f} Hz)")

    freq = Pest.freq
    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in freq])
    true_single = np.array([complex(P0(1j * 2 * np.pi * f)) for f in Pest_q.freq])

    fig1, _ = fdi.bode_fdi(
        [(Pest_q.freq, true_single), Pest_q],
        unc=(Pest_q.freq, Pest_q.userdata.sG[:, 0]),
        legend=["true", "FRF", "sG"], title="Single wideband experiment")
    save_fig(fig1, "tutorial_2_single.png")

    fig2, _ = fdi.bode_fdi(
        [(freq, true), Pest],
        unc=(freq, Pest.userdata.sG[:, 0]),
        legend=["true", "FRF (combined)", "sG"], title="Iterative (3 experiments)")
    save_fig(fig2, "tutorial_2_iterative.png")

    # parametric fit on the combined FRF
    n, mh, ml = 7, 4, 0
    SYS = {}
    SYS["ml"], SYS["ls"] = fdi.mlfdi(Pest, n, mh, ml, 500, 1e-10, 0, "c")
    SYS["btls"], SYS["gtls"] = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 500, 1e-10, "c")
    for name, Hm in SYS.items():
        fit = np.array([complex(control.tf(Hm[0, 0])(1j * 2 * np.pi * f)) for f in freq])
        print(f"  {name:5s} median rel err vs true: "
              f"{np.median(np.abs(fit - true) / np.abs(true)):.2e}")
    show()


if __name__ == "__main__":
    main()
