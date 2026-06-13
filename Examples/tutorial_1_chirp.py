"""Tutorial 1 (chirp) - Python port of ``Examples/Tutorial_1_chirp.m``.

Swept-sine (chirp) excitation: because the chirp is repeated periodically, the
FRF is estimated with the periodic H1 method (``time2frf_h1``), then a
parametric model is fitted.
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

    # STEP 1: swept-sine design
    harm = dict(fs=10000.0, df=1.0, fl=1.0, fh=1000.0)
    sweep = fdi.sweptsine(harm, dict(type="lin"))
    fs = harm["fs"]
    nrofs = sweep.x.size

    # experiment: 10 periods + noise
    rng = np.random.default_rng(0)
    u = np.tile(sweep.x * 1.6726, 10)
    t = np.arange(u.size) / fs
    u_n = u + 0.01 * rng.standard_normal(u.size)
    y = control.forced_response(P0, t, u_n).outputs
    y = y + 0.001 * rng.standard_normal(y.shape)

    # STEP 2: periodic H1 FRF
    x, _ = fdi.pretreat(u, nrofs, fs, 1, 0)
    y, _ = fdi.pretreat(y, nrofs, fs, 1, 0)
    _, _, FRF, freq, coh = fdi.time2frf_h1(x, y, fs, 1, 1000, 1, window=0)
    H = FRF[:, 0]

    # parametric fit
    n, mh, ml = 7, 4, 0
    W = coh[:, 0] * freq
    Hnls, _ = fdi.nlsfdi(H, freq, W, n, mh, ml, 500, 1e-8, 0, "c", fs)

    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in freq])
    fit = np.array([complex(control.tf(Hnls[0, 0])(1j * 2 * np.pi * f)) for f in freq])
    print(f"  fitted (n={n}) median rel err: "
          f"{np.median(np.abs(fit - true) / np.abs(true)):.2e}")

    fig, _ = fdi.bode_fdi([(freq, true), (freq, H), (freq, fit)],
                          labels=["true", "FRF (H1)", "NLS fit"],
                          title=f"Tutorial 1 chirp ({label})")
    save_fig(fig, "tutorial_1_chirp.png")
    show()


if __name__ == "__main__":
    main()
