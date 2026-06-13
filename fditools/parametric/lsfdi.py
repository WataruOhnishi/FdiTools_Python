"""Linear least-squares frequency-domain identification (port of ``lsfdi.m``).

Used to compute starting values for the iterative estimators.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..auxiliary.conversions import theta2ba, ba2hm, vectorize_orders
from ._residuals import waxis_of


def _as2d(a, nroff):
    a = np.atleast_2d(np.asarray(a))
    return a if a.shape[0] == nroff else a.T


def lsfdi(X, Y, freq, n, M_mh, M_ml, cORd, fs=None):
    """Return ``(Hls, waxis)`` — the LS transfer-function array and freq axis."""
    freq = np.asarray(freq, dtype=float).ravel()
    nroff = freq.size
    X = _as2d(X, nroff)
    Y = _as2d(Y, nroff)
    nrofi = X.shape[1]
    nrofo = Y.shape[1]
    nrofh = nrofi * nrofo

    mh = vectorize_orders(M_mh)
    ml = vectorize_orders(M_ml)
    nrofb = int(np.sum(mh - ml)) + nrofh

    waxis = waxis_of(freq, cORd, fs)
    if np.max(mh) > n:
        warnings.warn("numerator order is larger than denominator order")
    if np.min(mh - ml) < 0:
        raise ValueError("elements of M_ml must be smaller than those of M_mh")

    P = np.zeros((nrofh * nroff, n + 1), dtype=complex)
    Q = np.zeros((nrofh * nroff, nrofb), dtype=complex)
    index = 0
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        rows = slice(h * nroff, (h + 1) * nroff)
        P[rows, :] = (waxis[:, None] ** np.arange(n, -1, -1)) * Y[:, o][:, None]
        cnt = mh[h] - ml[h] + 1
        U = (waxis[:, None] ** np.arange(mh[h], ml[h] - 1, -1)) * X[:, i][:, None]
        Q[rows, index:index + cnt] = U
        index += cnt

    J = np.block([[np.real(P[:, 1:]), -np.real(Q)],
                  [np.imag(P[:, 1:]), -np.imag(Q)]])
    b = -np.concatenate([np.real(P[:, 0]), np.imag(P[:, 0])])
    y = np.linalg.pinv(J) @ b
    Bls, Als = theta2ba(y, n, mh, ml)
    Hls = ba2hm(Bls, Als, nrofi, nrofo)
    return Hls, waxis
