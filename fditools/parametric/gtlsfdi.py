"""Generalized total least-squares FDI (port of ``gtlsfdi.m``).

Note: the original ``try chol(A) ... catch`` block references an undefined ``A``
and therefore *always* falls through to the ``catch`` branch.  That behaviour is
reproduced here (only the (1,1) block is Cholesky-factored) so the Python port
matches the MATLAB results.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from ..auxiliary.conversions import theta2ba, ba2hm, vectorize_orders
from ._residuals import waxis_of, qsvd


def _as2d(a, nroff, dtype):
    a = np.atleast_2d(np.asarray(a, dtype=dtype))
    return a if a.shape[0] == nroff else a.T


def _chol_upper(M):
    """MATLAB ``chol`` (upper R with R'R = M), with a tiny jitter fallback."""
    M = (M + M.conj().T) / 2.0
    try:
        return sla.cholesky(M, lower=False)
    except sla.LinAlgError:
        d = np.real(np.diag(M))
        jit = (np.max(np.abs(d)) + 1.0) * 1e-12
        return sla.cholesky(M + jit * np.eye(M.shape[0]), lower=False)


def _chol_block(MytMy, MytMx, MxtMx):
    """Reproduce the (buggy) ``catch`` branch: chol only the (1,1) block."""
    R = _chol_upper(MytMy)
    return np.block([[R, MytMx], [MytMx.conj().T, MxtMx]])


def _solve_qsvd(J, C, nrofp):
    Xg = qsvd(J, C)
    Xg = np.linalg.inv(Xg.conj().T)
    col = Xg[:, nrofp] / Xg[0, nrofp]
    return col


def gtlsfdi(X, Y, freq, n, M_mh, M_ml, sX2, sY2, cXY, cORd, fs=None):
    """Return ``(Hgtls, waxis)``."""
    freq = np.asarray(freq, dtype=float).ravel()
    nroff = freq.size
    X = _as2d(X, nroff, complex)
    Y = _as2d(Y, nroff, complex)
    sX2 = _as2d(sX2, nroff, float)
    sY2 = _as2d(sY2, nroff, float)
    cXY = _as2d(cXY, nroff, complex)
    nrofi, nrofo = X.shape[1], Y.shape[1]
    nrofh = nrofi * nrofo

    mh = vectorize_orders(M_mh)
    ml = vectorize_orders(M_ml)
    nrofb = int(np.sum(mh - ml)) + nrofh
    nrofp = nrofb + n

    waxis = waxis_of(freq, cORd, fs)
    if np.max(mh) > n:
        warnings.warn("numerator order is larger than denominator order")

    # weighted Jacobian (SEr = 1 for GTLS)
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
    J = np.block([[np.real(P), -np.real(Q)], [np.imag(P), -np.imag(Q)]])

    C = np.zeros((nrofh * (nrofp + 1), nrofp + 1), dtype=complex)
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        cnt = mh[h] - ml[h] + 1
        MytMy = np.zeros((n + 1, n + 1), dtype=complex)
        MxtMx = np.zeros((cnt, cnt), dtype=complex)
        MytMx = np.zeros((n + 1, cnt), dtype=complex)
        for p in range(n, -1, -1):
            for q in range(n, -1, -1):
                MytMy[n - p, n - q] = 2.0 * np.real(((-1) ** q)
                    * np.sum(waxis ** (p + q) * sY2[:, o]))
        for p in range(mh[h], ml[h] - 1, -1):
            for q in range(mh[h], ml[h] - 1, -1):
                MxtMx[mh[h] - p, mh[h] - q] = 2.0 * np.real(((-1) ** q)
                    * np.sum(waxis ** (p + q) * sX2[:, i]))
        for p in range(n, -1, -1):
            for q in range(mh[h], ml[h] - 1, -1):
                MytMx[n - p, mh[h] - q] = -2.0 * np.real(((-1) ** q)
                    * np.sum(waxis ** (p + q) * cXY[:, h]))
        block = _chol_block(MytMy, MytMx, MxtMx)
        r0 = h * (nrofp + 1)
        C[r0:r0 + (n + 1 + cnt), :(n + 1 + cnt)] = block

    col = _solve_qsvd(J, C, nrofp)
    Bgtls, Agtls = theta2ba(col[1:], n, mh, ml)
    Hgtls = ba2hm(Bgtls, Agtls, nrofi, nrofo)
    return Hgtls, waxis
