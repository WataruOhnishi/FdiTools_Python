"""LPM tutorial (v3.0) - Local Polynomial Method FRF estimation.

Shows the strength of ``time2frf_lpm``: on a SHORT record whose start-up
transient is **not** removed, the LPM models the transient and still returns a
low-bias FRF, whereas the plain ML estimate (no transient removal) is biased by
leakage.  Falls back to a synthetic plant if the benchmark model is absent.
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
    ms = fdi.multisine(harm, control.tf([1], [1]),
                       dict(itp="r", ctp="c", dtp="f", gtp="l"))
    fs = harm["fs"]

    # short record (4 periods) WITH the start-up transient kept in
    u = np.tile(np.squeeze(ms.x[0, 0, :]), 4)
    t = np.arange(u.size) / fs
    y = control.forced_response(P0, t, u).outputs
    y = y + 1e-4 * np.random.default_rng(0).standard_normal(y.shape)

    freq_true = ms.freq[ms.ex]
    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in freq_true])

    # LPM on the raw data (models the transient)
    FRF, freq, sG, T = fdi.time2frf_lpm(u, y, fs, order=2, halfwidth=3,
                                        period=ms.nrofs, lines=ms.ex)
    rel_lpm = np.median(np.abs(FRF[:, 0] - true) / np.abs(true))

    # plain ML on the SAME raw data (no transient removal -> biased)
    Pml = fdi.time2frf_ml(u, y, ms)
    rel_ml = np.median(np.abs(Pml.response[0, 0, :] - true) / np.abs(true))

    print(f"LPM (raw, transient kept) median rel err: {rel_lpm:.2e}")
    print(f"ML  (raw, no pretreat)    median rel err: {rel_ml:.2e}")

    fig, _ = fdi.bode_fdi(
        [(freq_true, true), (freq, FRF[:, 0]), Pml],
        legend=["true", "LPM (raw)", "ML (raw, biased)"],
        title=f"LPM vs ML on a short transient-corrupted record ({label})")
    save_fig(fig, "tutorial_lpm.png")
    show()


if __name__ == "__main__":
    main()
