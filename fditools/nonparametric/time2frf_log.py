"""Logarithmic averaging estimation of the FRF (port of ``time2frf_log.m``).

For non-synchronised / overlapping (Welch) measurements.  ``fl``/``fh`` are
frequency-line (bin) numbers as in the original.
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


def time2frf_log(x, y, fs, fl, fh, df, window=0, nrofl=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    m = x.shape[1]
    nroft, l = y.shape
    nrofs = int(round(fs / df))
    freq = np.arange(fl, fh + 1) * df
    nline = fh - fl + 1

    win = _window(window, nrofs)

    k = int((nroft - nrofl) // (nrofs - nrofl))
    X = np.zeros((nline, m * k), dtype=complex)
    Y = np.zeros((nline, l * k), dtype=complex)
    start = 0
    for i in range(k):
        idx = slice(start, start + nrofs)
        Xw = np.fft.fft(x[idx, :] * win[:, None], axis=0)
        Yw = np.fft.fft(y[idx, :] * win[:, None], axis=0)
        X[:, i * m:(i + 1) * m] = Xw[fl:fh + 1, :]
        Y[:, i * l:(i + 1) * l] = Yw[fl:fh + 1, :]
        start += nrofs - nrofl

    FRF = np.zeros((nline, m * l), dtype=complex)
    coh = np.zeros((nline, l))
    nwin = k - m + 1
    for fk in range(nline):
        Hi = np.zeros((l, m), dtype=complex)
        Xi = Yi = None
        for i in range(nwin):
            Xi = np.array([X[fk, (ii + i) * m:(ii + i + 1) * m] for ii in range(m)]).T
            Yi = np.array([Y[fk, (ii + i) * l:(ii + i + 1) * l] for ii in range(m)]).T
            Hi = Hi + Yi @ np.linalg.inv(Xi)
        Hi = np.unwrap(np.angle(Hi / nwin))
        Hav = np.zeros((l, m), dtype=complex)
        for i in range(nwin):
            Xi = np.array([X[fk, (ii + i) * m:(ii + i + 1) * m] for ii in range(m)]).T
            Yi = np.array([Y[fk, (ii + i) * l:(ii + i + 1) * l] for ii in range(m)]).T
            Hav = Hav + np.log(Yi @ np.linalg.inv(Xi) * np.exp(-1j * Hi))
        Hav = np.exp(Hav / nwin) * np.exp(1j * Hi)
        Hav_col = Hav.T.flatten()
        FRF[fk, :] = Hav_col
        XiXi = Xi @ Xi.conj().T
        YiYi = Yi @ Yi.conj().T
        for i in range(l):
            hi = Hav[i, :]
            coh[fk, i] = np.real((hi @ XiXi @ hi.conj().T) / YiYi[i, i])
    return X, Y, FRF, freq, coh
