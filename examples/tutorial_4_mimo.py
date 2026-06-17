"""MIMO tutorial (v3.0) - Python port of ``Examples/Tutorial_4_MIMO.m`` (core).

Identifies a 2x2 modal plant from an orthogonal 2-input multisine:
  * non-parametric MIMO FRF with ``time2frf_ml`` (and ``time2frf_lpm``),
  * structured modal parameters with ``frf2modal``.
Self-contained synthetic 2x2 plant (the benchmark model is SISO).
"""

import numpy as np
import control
import matplotlib.pyplot as plt

import fditools as fdi
from _plot import save_fig, show


def main():
    # --- 2x2 modal plant (two resonances, with cross-coupling) ----------
    w1, w2 = 2 * np.pi * 60, 2 * np.pi * 150
    g = [[control.tf([w1 ** 2], [1, 2 * 0.02 * w1, w1 ** 2]),
          control.tf([0.3 * w1 ** 2], [1, 2 * 0.05 * w1, w1 ** 2])],
         [control.tf([0.2 * w2 ** 2], [1, 2 * 0.04 * w2, w2 ** 2]),
          control.tf([w2 ** 2], [1, 2 * 0.03 * w2, w2 ** 2])]]
    nu = ny = 2

    # --- orthogonal 2-input multisine -> 2 experiments ------------------
    harm = dict(fs=5000.0, df=1.0, fl=5.0, fh=400.0, fr=1.02)
    ms = fdi.multisine(harm, [control.tf([1], [1]), control.tf([1], [1])],
                       dict(itp="r", ctp="n", dtp="f", gtp="l"))
    ne = ms.x.shape[1]
    nrofp = 6
    N = nrofp * ms.nrofs
    T = np.arange(N) / harm["fs"]
    rng = np.random.default_rng(0)

    Xl, Yl = [], []          # transient-removed (for ML)
    Xraw, Yraw = [], []      # raw (for LPM)
    for e in range(ne):
        Ue = np.tile(ms.x[:, e, :].T, (nrofp, 1))
        Ye = np.zeros((N, ny))
        for o in range(ny):
            for i in range(nu):
                Ye[:, o] += control.forced_response(g[o][i], T, Ue[:, i]).outputs
            Ye[:, o] += 1e-4 * rng.standard_normal(N)
        Xraw.append(Ue)
        Yraw.append(Ye)
        Xl.append(fdi.pretreat(Ue, ms.nrofs, harm["fs"], 1, 0)[0])
        Yl.append(fdi.pretreat(Ye, ms.nrofs, harm["fs"], 1, 0)[0])

    # --- non-parametric MIMO FRF ----------------------------------------
    Pml = fdi.time2frf_ml(np.stack(Xl, 2), np.stack(Yl, 2), ms)
    Plpm = fdi.time2frf_lpm(np.stack(Xraw, 2), np.stack(Yraw, 2), harm["fs"],
                            order=2, halfwidth=3, period=ms.nrofs, lines=ms.ex)
    print(f"MIMO FRF: {Pml.response.shape}  method={Pml.userdata.method}")

    freq = Pml.freq
    true = np.array([[[complex(g[o][i](1j * 2 * np.pi * f)) for f in freq]
                      for i in range(nu)] for o in range(ny)])
    print("ML  median rel err:", f"{np.median(np.abs(Pml.response - true) / np.abs(true)):.2e}")
    print("LPM median rel err:", f"{np.median(np.abs(Plpm.response - true) / np.abs(true)):.2e}")

    # --- structured modal identification --------------------------------
    modal, Pm = fdi.frf2modal(Pml, 0, 2, damping="proportional",
                              initfreq=[60, 150], initdamp=0.02, feedthrough=False)
    print("modal wn [Hz]:", np.round(np.sort(modal["wn"]), 2), " (true [60, 150])")
    print("modal zeta   :", np.round(modal["zeta"][np.argsort(modal["wn"])], 4),
          " (true [0.02, 0.03])")

    # --- plot the (1,1) channel -----------------------------------------
    modal_resp = np.array([complex(control.tf(control.ss(Pm)[0, 0])(1j * 2 * np.pi * f))
                           for f in freq])
    fig, _ = fdi.bode_fdi(
        [(freq, true[0, 0]), Pml[0, 0], (freq, modal_resp)],
        legend=["true", "FRF (ML)", "modal fit"],
        title="MIMO 2x2 identification - channel (1,1)")
    save_fig(fig, "tutorial_4_mimo.png")
    show()


if __name__ == "__main__":
    main()
