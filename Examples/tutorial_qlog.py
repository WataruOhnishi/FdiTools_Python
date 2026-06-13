"""Quasi-log multisine identification tutorial (Python port of
``Examples/Tutorial_1_qlog.m``).

Self-contained: a synthetic two-resonance plant stands in for the benchmark
``.mat`` model.  Run with matplotlib installed to see the Bode plots::

    python examples/tutorial_qlog.py
"""

import numpy as np
import control

import fditools as fdi


def true_plant():
    """A 4th-order plant with two lightly damped resonances."""
    g1 = control.tf([(2 * np.pi * 90) ** 2],
                    [1, 2 * 0.02 * 2 * np.pi * 90, (2 * np.pi * 90) ** 2])
    g2 = control.tf([(2 * np.pi * 240) ** 2],
                    [1, 2 * 0.015 * 2 * np.pi * 240, (2 * np.pi * 240) ** 2])
    return control.minreal(g1 * g2)


def main(show=True):
    P0 = true_plant()

    # --- STEP 1: excitation design (quasi-log multisine) ----------------
    harm = dict(fs=10000.0, df=1.0, fl=1.0, fh=1000.0, fr=1.02)
    options = dict(itp="r", ctp="c", dtp="f", gtp="q")
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    print(f"multisine: {ms.ex.size} excited lines, CF = {ms.cf[0, 0]:.3f}")

    # --- experiment: periodic excitation through the plant --------------
    one = np.squeeze(ms.x[0, 0, :])
    nrofp = 5
    u = np.tile(one, nrofp)
    u = u + 0.01 * np.random.default_rng(0).standard_normal(u.size)
    T = np.arange(u.size) / harm["fs"]
    y = control.forced_response(P0, T, u).outputs
    y = y + 0.001 * np.random.default_rng(1).standard_normal(y.shape)

    # --- STEP 2: non-parametric FRF -------------------------------------
    xp, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
    yp, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
    Pest = fdi.time2frf_ml(xp, yp, ms)

    # --- STEP 4: parametric estimation ----------------------------------
    n, mh, ml = 4, 2, 0
    FRF_W = np.ones(Pest.freq.size)
    SYS = {}
    SYS["nls"], SYS["wls"] = fdi.nlsfdi(Pest, FRF_W, n, mh, ml, 500, 1e-10, 0, "c")
    SYS["ml"], SYS["ls"] = fdi.mlfdi(Pest, n, mh, ml, 500, 1e-10, 0, "c")
    SYS["btls"], SYS["gtls"] = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 500, 1e-10, "c")

    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in Pest.freq])
    for name, Hm in SYS.items():
        sys = control.tf(Hm[0, 0])
        fit = np.array([complex(sys(1j * 2 * np.pi * f)) for f in Pest.freq])
        err = np.median(np.abs(fit - true) / np.abs(true))
        print(f"{name:5s} median rel err vs true: {err:.2e}")

    if show:
        try:
            import matplotlib.pyplot as plt

            fdi.bode_fdi(
                [(Pest.freq, true), Pest, fdi.FrfData(
                    fdi.hfrf(SYS["btls"], Pest.freq), Pest.freq)],
                noise=(Pest.freq, Pest.userdata.sGhat[:, 0]),
                labels=["true", "FRF", "BTLS"],
                title="Quasi-log multisine identification",
            )
            plt.show()
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
