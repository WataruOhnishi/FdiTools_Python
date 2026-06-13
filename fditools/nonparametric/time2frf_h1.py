"""H1 estimation of the FRF with windowing (port of ``time2frf_h1.m``).

As in the original, ``fl``/``fh`` are frequency-line (bin) numbers; the toolbox
examples use ``df = 1`` so they coincide with Hz.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows


def _window(kind, n):
    if kind == 0:
        return np.ones(n)
    if kind == 1:
        return windows.hann(n, sym=True)
    if kind == 2:
        return windows.chebwin(n, at=100)
    if kind == 3:
        return windows.bartlett(n)
    raise ValueError(f"unknown window {kind}")


def time2frf_h1(x, y, fs, fl, fh, df, window=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    m = x.shape[1]
    nroft, l = y.shape
    nrofs = int(round(fs / df))
    nrofp = nroft // nrofs
    freq = fs * np.arange(fl / df, fh / df + 1) / nrofs

    win = _window(window, nrofs)
    nline = fh - fl + 1

    X = np.zeros((nline, m * nrofp), dtype=complex)
    Y = np.zeros((nline, l * nrofp), dtype=complex)
    for k in range(nrofp):
        Xk = np.fft.fft(x[k * nrofs:(k + 1) * nrofs, :] * win[:, None], axis=0)
        Yk = np.fft.fft(y[k * nrofs:(k + 1) * nrofs, :] * win[:, None], axis=0)
        X[:, k * m:(k + 1) * m] = Xk[fl:fh + 1, :]
        Y[:, k * l:(k + 1) * l] = Yk[fl:fh + 1, :]

    coh = np.zeros((nline, l))
    FRF = np.zeros((nline, m * l), dtype=complex)
    X2 = np.zeros((m * nline, m), dtype=complex)
    Y2 = np.zeros((l * nline, l), dtype=complex)
    for k in range(nline):
        Xk = np.array([X[k, i * m:(i + 1) * m] for i in range(nrofp)]).T  # (m, nrofp)
        Yk = np.array([Y[k, i * l:(i + 1) * l] for i in range(nrofp)]).T  # (l, nrofp)
        XkXk = Xk @ Xk.conj().T
        YkYk = Yk @ Yk.conj().T
        H1 = (Yk @ Xk.conj().T) @ np.linalg.inv(XkXk)   # (l, m)
        FRF[k, :] = H1.T.flatten()                       # [H11..H1m H21..]
        for i in range(l):
            coh[k, i] = np.real((H1[i, :] @ XkXk @ H1[i, :].conj().T) / YkYk[i, i])
        X2[m * k:m * (k + 1), :] = XkXk
        Y2[l * k:l * (k + 1), :] = YkYk
    return X2, Y2, FRF, freq, coh
