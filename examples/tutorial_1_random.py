"""Tutorial 1 (random) - Python port of ``Examples/Tutorial_1_random.m``.

Conventional random-noise excitation: estimate the FRF with Welch averaging
(SciPy's equivalent of MATLAB ``tfestimate``/``mscohere``) and fit a parametric
model.  MATLAB uses ``tfest`` (System Identification Toolbox); here we use the
toolbox's own ``nlsfdi`` on the measured FRF.
"""

import numpy as np
import control
import matplotlib.pyplot as plt
from scipy.signal import csd, welch, coherence, detrend

import fditools as fdi
from _data import benchmark_plant
from _plot import save_fig, show


def welch_frf(u, y, fs, nperseg):
    """H1 FRF estimate and coherence (matches MATLAB tfestimate/mscohere)."""
    f, Pyx = csd(y, u, fs=fs, window="boxcar", nperseg=nperseg, noverlap=0)
    _, Pxx = welch(u, fs=fs, window="boxcar", nperseg=nperseg, noverlap=0)
    _, cxy = coherence(u, y, fs=fs, window="boxcar", nperseg=nperseg, noverlap=0)
    return f, Pyx / Pxx, cxy


def main():
    P0, label = benchmark_plant()
    print(f"true plant: {label}")

    fs, texp, amp = 10000.0, 5.0, 1.6726
    rng = np.random.default_rng(0)
    u = (rng.random(int(fs * texp)) * 2 - 1) * amp

    t = np.arange(u.size) / fs
    u_n = u + 0.01 * rng.standard_normal(u.size)
    y = control.forced_response(P0, t, u_n).outputs
    y = y + 0.001 * rng.standard_normal(y.shape)

    u = detrend(u, type="constant")
    y = detrend(y, type="constant")

    freq, H, cxy = welch_frf(u, y, fs, nperseg=int(fs))   # ~1 Hz resolution

    # restrict to the excitation band and fit a parametric model
    band = (freq >= 1) & (freq <= 1000)
    fb, Hb = freq[band], H[band]
    n, mh, ml = 7, 4, 0
    W = (cxy[band] * fb)                                   # coherence weighting
    Hnls, _ = fdi.nlsfdi(Hb, fb, W, n, mh, ml, 500, 1e-8, 0, "c", fs)

    true = np.array([complex(P0(1j * 2 * np.pi * f)) for f in fb])
    fit = np.array([complex(control.tf(Hnls[0, 0])(1j * 2 * np.pi * f)) for f in fb])
    print(f"  fitted (n={n}) median rel err: "
          f"{np.median(np.abs(fit - true) / np.abs(true)):.2e}")

    fig, _ = fdi.bode_fdi([(fb, true), (fb, Hb), (fb, fit)],
                          legend=["true", "FRF (H1)", "NLS fit"],
                          title=f"Tutorial 1 random ({label})")
    save_fig(fig, "tutorial_1_random.png")

    figc, ax = plt.subplots(figsize=(7, 3))
    ax.semilogx(freq, cxy)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence [-]")
    ax.set_xlim(1, 1000)
    save_fig(figc, "tutorial_1_random_coh.png")
    show()


if __name__ == "__main__":
    main()
