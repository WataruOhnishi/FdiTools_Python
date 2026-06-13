"""Weighted least-squares FDI with Sanathanan-Koerner reweighting
(port of ``wlsfdi.m``)."""

from __future__ import annotations

import warnings

import numpy as np

from ..aux.conversions import theta2ba, ba2hm, vectorize_orders
from ._residuals import waxis_of


def _coerce_frf(FRF, nroff, nrofh):
    FRF = np.atleast_2d(np.asarray(FRF, dtype=complex))
    if FRF.shape[0] != nroff:
        FRF = FRF.T
    if FRF.shape[1] != nrofh:
        FRF = FRF.reshape(nroff, nrofh)
    return FRF


def wlsfdi(FRF, freq, FRF_W, n, M_mh, M_ml, cORd, fs=None):
    """Return ``(Hwls, waxis)``."""
    freq = np.asarray(freq, dtype=float).ravel()
    nroff = freq.size

    mh_mat = np.atleast_2d(np.asarray(M_mh))
    nrofo, nrofi = mh_mat.shape
    nrofh = nrofi * nrofo
    mh = vectorize_orders(M_mh)
    ml = vectorize_orders(M_ml)
    nrofb = int(np.sum(mh - ml)) + nrofh

    FRF = _coerce_frf(FRF, nroff, nrofh)
    FRF_W = _coerce_frf(FRF_W, nroff, nrofh).real.astype(float)

    waxis = waxis_of(freq, cORd, fs)
    if np.max(mh) > n:
        warnings.warn("numerator order is larger than denominator order")
    if np.min(mh - ml) < 0:
        raise ValueError("elements of M_ml must be smaller than those of M_mh")

    Bwls = Awls = None
    for _step in range(2):
        FRF_WD = FRF * FRF_W
        P = np.zeros((nrofh * nroff, n + 1), dtype=complex)
        Q = np.zeros((nrofh * nroff, nrofb), dtype=complex)
        index = 0
        for h in range(nrofh):
            rows = slice(h * nroff, (h + 1) * nroff)
            P[rows, :] = (waxis[:, None] ** np.arange(n, -1, -1)) * FRF_WD[:, h][:, None]
            cnt = mh[h] - ml[h] + 1
            U = (waxis[:, None] ** np.arange(mh[h], ml[h] - 1, -1)) * FRF_W[:, h][:, None]
            Q[rows, index:index + cnt] = U
            index += cnt

        J = np.block([[np.real(P[:, 1:]), -np.real(Q)],
                      [np.imag(P[:, 1:]), -np.imag(Q)]])
        b = -np.concatenate([np.real(P[:, 0]), np.imag(P[:, 0])])
        y = np.linalg.pinv(J) @ b
        Bwls, Awls = theta2ba(y, n, mh, ml)

        # Sanathanan-Koerner weighting update
        P_fr = waxis[:, None] ** np.arange(n, -1, -1)
        Ajw = P_fr @ Awls
        FRF_W = FRF_W / np.abs(Ajw)[:, None]

    Hwls = ba2hm(Bwls, Awls, nrofi, nrofo)
    return Hwls, waxis
