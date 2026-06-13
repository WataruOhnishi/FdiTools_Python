"""Small numeric helpers (port of ``f2t``/``t2f``/``dbm``/``phs``)."""

from __future__ import annotations

import numpy as np


def _next_pow2_len(n):
    N = 2
    while n > N:
        N *= 2
    return N


def f2t(X, N=None):
    """Time signals from Fourier coefficients: ``x = N*real(ifft(X, N))``."""
    X = np.asarray(X)
    if N is None:
        nrow = X.shape[0]
        N = _next_pow2_len(2 * nrow)
    x = N * np.real(np.fft.ifft(X, n=N, axis=0))
    return x


def t2f(x, N=None):
    """Fourier-series coefficients from time signals.

    ``X = [X(1); 2*X(2:floor(N/2))]/N`` (single-sided, DC un-doubled).
    """
    x = np.asarray(x)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    if N is None:
        N = _next_pow2_len(max(x.shape))
    Xf = np.fft.fft(x, n=N, axis=0)
    half = int(np.floor(N / 2))
    X = np.vstack([Xf[0:1, :], 2.0 * Xf[1:half, :]]) / N
    if one_d:
        X = X[:, 0]
    return X


def dbm(X):
    """Magnitude in decibels: ``20*log10(abs(X))``."""
    return 20.0 * np.log10(np.abs(np.asarray(X)))


def phs(X, shift_removal=True, glitch=None):
    """Phase in degrees with display-oriented shift/glitch removal.

    Port of ``phs.m``.  ``X`` is a complex vector; the returned real vector is
    the wrapped phase (deg) cleaned of 360-degree up/down jump patterns, and –
    when *glitch* (a threshold in degrees) is given – of isolated spikes.
    """
    X = np.asarray(X).ravel()
    Y = np.angle(X) * 180.0 / np.pi
    n = Y.size
    shift = 160.0

    if shift_removal:
        j = 0
        first = 0
        # 1-based MATLAB loop i = 2..n  ->  python idx 1..n-1
        i = 1
        while i < n:
            jump_i = 0
            jump_j = 0
            end_j = 0
            bgn_j = 0
            if abs(Y[i] - Y[i - 1]) > shift:
                if j != 0:
                    jump_i = 1 if Y[i] > Y[i - 1] else -1
                else:
                    bgn_j = 1
                # advance j until next big jump (MATLAB j=i..n-1, 1-based)
                j = i
                while j < n - 1:
                    if abs(Y[j] - Y[j + 1]) > shift:
                        break
                    j += 1
                if j != n - 1:
                    jump_j = -1 if Y[j] > Y[j + 1] else 1
                else:
                    end_j = 1
                if bgn_j == 1 and first == 0:
                    if Y[i] < Y[i - 1]:
                        Y[:i] = Y[:i] - 360.0
                    else:
                        jump_i = 1
                    first = 1
                if jump_i == -jump_j:
                    Y[i:j + 1] = Y[i:j + 1] - 360.0
                if end_j == 1:
                    pass
            i += 1

    if glitch is not None:
        for i in range(2, n - 2):
            if ((abs(Y[i] - Y[i - 1]) > glitch or abs(Y[i] - Y[i - 2]) > glitch)
                    and (abs(Y[i] - Y[i + 1]) > glitch or abs(Y[i] - Y[i + 2]) > glitch)):
                Y[i] = np.mean([Y[i - 3], Y[i - 2], Y[i - 1],
                                Y[i + 1], Y[i + 2], Y[i + 3]])
    return Y
