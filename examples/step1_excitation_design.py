"""Step 1 - Excitation design (Python port of ``Examples/Step_1_ExcitationDesign.m``).

Demonstrates the three excitation signals: multisine, PRBS and swept sine.
Needs no measurement data.  Run with the ``.venv`` interpreter selected::

    python examples/step1_excitation_design.py
"""

import numpy as np
import control
import matplotlib.pyplot as plt

import fditools as fdi
from _plot import save_fig, show


def main():
    # ----- Example 1: multisine -----------------------------------------
    harm = dict(fs=1000.0, df=1.0, fl=1.0, fh=100.0, fr=1.02)
    options = dict(itp="r", ctp="c", dtp="f", gtp="l")
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    print(f"multisine: crest factor CF = {ms.cf[0, 0]:.3f}")

    fig1, (a, b) = plt.subplots(2, 1, figsize=(7, 5))
    a.plot(ms.time, np.squeeze(ms.x))
    a.set_title(f"Multisine (CF = {ms.cf[0, 0]:.3f})")
    a.set_xlabel("time [s]")
    a.set_ylabel("amplitude [-]")
    b.semilogx(ms.freq, fdi.dbm(np.squeeze(ms.X)), "+")
    b.set_xlabel("frequency [Hz]")
    b.set_ylabel("amplitude [dB]")
    fig1.suptitle("Example 1: Multisine")
    save_fig(fig1, "step1_multisine.png")

    # ----- Example 2: PRBS ----------------------------------------------
    fs, df = 1000.0, 1.0
    nrofs = int(fs / df)
    time = np.arange(nrofs) / fs
    log2N = 21
    x, X, freq, _ = fdi.prbs(fs, log2N, bitno=nrofs)

    fig2, (a, b) = plt.subplots(2, 1, figsize=(7, 5))
    a.plot(time, x)
    a.set_title("PRBS: time domain")
    a.set_xlabel("time [s]")
    a.set_ylabel("amplitude [-]")
    a.set_ylim(-2, 2)
    b.semilogx(freq, fdi.dbm(X))
    b.set_title("PRBS: frequency domain")
    b.set_xlabel("frequency [Hz]")
    b.set_ylabel("amplitude [dB]")
    save_fig(fig2, "step1_prbs.png")

    # ----- Example 3: swept sine ----------------------------------------
    h = dict(fs=2000.0, df=0.05, fl=0.1, fh=100.0)
    sweep = fdi.sweptsine(h, dict(type="lin"))

    fig3, (a, b) = plt.subplots(2, 1, figsize=(7, 5))
    a.plot(sweep.time, sweep.x)
    a.set_title("Swept-sine: time domain")
    a.set_xlabel("time [s]")
    a.set_ylabel("amplitude [-]")
    b.semilogx(sweep.freq, fdi.dbm(sweep.X))
    b.set_title("Swept-sine: frequency domain")
    b.set_xlabel("frequency [Hz]")
    b.set_ylabel("amplitude [dB]")
    save_fig(fig3, "step1_sweptsine.png")

    show()


if __name__ == "__main__":
    main()
