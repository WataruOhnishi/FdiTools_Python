"""Tutorial 1 (qlog) - Python port of ``MATLAB/Examples/Tutorial_1_qlog.m``.

Quasi-log multisine identification of the benchmark plant (``mdl.Pv(1,1)``).
Falls back to a synthetic plant if the model has not been converted yet
(see ``MATLAB/Examples/private/convert_ident_to_python.m``).
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

    # STEP 1: excitation design (quasi-log multisine)
    harm = dict(fs=10000.0, df=1.0, fl=1.0, fh=1000.0, fr=1.02)
    options = dict(itp="r", ctp="c", dtp="f", gtp="q")
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    print(f"multisine: {ms.ex.size} excited lines, CF = {ms.cf[0, 0]:.3f}")

    # experiment: 5 periods + measurement noise
    rng = np.random.default_rng(0)
    u = np.tile(np.squeeze(ms.x[0, 0, :]), 5)
    u = u + 0.01 * rng.standard_normal(u.size)
    t = np.arange(u.size) / harm["fs"]
    y = control.forced_response(P0, t, u).outputs
    y = y + 0.001 * rng.standard_normal(y.shape)

    # STEP 2: non-parametric FRF
    x, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
    y, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
    Pest = fdi.time2frf_ml(x, y, ms)
    freq = Pest.freq

    # STEP 4: parametric estimation
    n, mh, ml = 7, 4, 0
    FRF_W = np.ones(freq.size)
    SYS = {}
    SYS["nls"], SYS["wls"] = fdi.nlsfdi(Pest, FRF_W, n, mh, ml, 500, 1e-10, 0, "c")
    SYS["ml"], SYS["ls"] = fdi.mlfdi(Pest, n, mh, ml, 500, 1e-10, 0, "c")
    SYS["btls"], SYS["gtls"] = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 500, 1e-10, "c")

    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in freq])
    for name, Hm in SYS.items():
        sys = control.tf(Hm[0, 0])
        fit = np.array([complex(sys(1j * 2 * np.pi * f)) for f in freq])
        print(f"  {name:5s} median rel err vs true: "
              f"{np.median(np.abs(fit - true) / np.abs(true)):.2e}")

    fig, _ = fdi.bode_fdi(
        [(freq, true), Pest,
         fdi.FrfData(fdi.hfrf(SYS["btls"], freq), freq)],
        unc=(freq, Pest.userdata.sG[:, 0]),
        legend=["true", "FRF", "BTLS", "sG"],
        title=f"Tutorial 1 qlog ({label})")
    save_fig(fig, "tutorial_1_qlog.png")
    show()


if __name__ == "__main__":
    main()
