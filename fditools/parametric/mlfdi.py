"""Maximum-likelihood FDI (Gauss-Newton / Levenberg-Marquardt) — port of
``mlfdi.m``.

Structured : ``mlfdi(Pest, n, mh, ml, iterno, relvar, GN, cORd)``
Classical  : ``mlfdi(X, Y, freq, sX2, sY2, cXY, n, mh, ml, iterno, relvar, GN, cORd, fs)``
Returns ``(Hml, Hls)``.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData
from ..auxiliary.conversions import theta2ba, ba2theta, ba2hm, hm2ba, vectorize_orders
from .lsfdi import lsfdi
from ._residuals import mlfdi_res


def _as2d(a, nroff, dtype=complex):
    a = np.atleast_2d(np.asarray(a, dtype=dtype))
    return a if a.shape[0] == nroff else a.T


def mlfdi(*args, verbose=False):
    if isinstance(args[0], FrfData):
        Pest, n, M_mh, M_ml, iterno, relvar, GN, cORd = args[:8]
        ud = Pest.userdata
        X, Y = ud.X, ud.Y
        freq = Pest.freq
        sX2, sY2, cXY = ud.sX2, ud.sY2, ud.cXY
        ms = ud.ms
        fs = (ms[0] if isinstance(ms, (list, tuple)) else ms).harm.fs
    else:
        (X, Y, freq, sX2, sY2, cXY, n, M_mh, M_ml,
         iterno, relvar, GN, cORd, fs) = args[:14]

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

    Hls, waxis = lsfdi(X, Y, freq, n, mh, ml, cORd, fs)

    relax = 0.0 if GN == 1 else 1.0
    Bml, Aml = hm2ba(Hls)
    iter0 = 0
    relerror0 = np.inf
    relerror = np.inf
    y = ba2theta(Bml, Aml, n, mh, ml)
    cost0 = mlfdi_res(Bml, Aml, freq, X, Y, sX2, sY2, cXY, waxis)

    P = waxis[:, None] ** np.arange(n, -1, -1)
    Q = waxis[:, None] ** np.arange(int(np.max(mh)), int(np.min(ml)) - 1, -1)

    it = 0
    while it < iterno and relerror > relvar:
        it += 1
        Num = P @ Bml.T            # (nroff, nrofh)
        Den = P @ Aml              # (nroff,)

        E = np.zeros(nrofh * nroff, dtype=complex)
        SE = np.zeros(nrofh * nroff)
        dA = np.zeros((nrofh * nroff, n), dtype=complex)
        dB = np.zeros((nrofh * nroff, nrofb), dtype=complex)
        index = 0
        for h in range(nrofh):
            i = h // nrofo
            o = h - i * nrofo
            rows = slice(h * nroff, (h + 1) * nroff)
            se = np.sqrt(sX2[:, i] * np.abs(Num[:, h]) ** 2
                         + sY2[:, o] * np.abs(Den) ** 2
                         - 2.0 * np.real(cXY[:, h] * Den * np.conj(Num[:, h])))
            eb = Num[:, h] * X[:, i] - Den * Y[:, o]
            SE[rows] = se
            E[rows] = eb
            for jj in range(n):
                Pj = P[:, jj + 1]
                WW = (-Y[:, o] * Pj / se
                      - eb / se ** 3 * (sY2[:, o] * np.real(Den * np.conj(Pj))
                                        - np.real(cXY[:, h] * Pj * np.conj(Num[:, h]))))
                dA[rows, jj] = WW
            cnt = mh[h] - ml[h] + 1
            for jc in range(cnt):
                Qj = Q[:, jc]
                WW = (X[:, i] * Qj / se
                      - eb / se ** 3 * (sX2[:, i] * np.real(Num[:, h] * np.conj(Qj))
                                        - np.real(cXY[:, h] * Den * np.conj(Qj))))
                dB[rows, index + jc] = WW
            index += cnt

        J = np.block([[np.real(dA), np.real(dB)],
                      [np.imag(dA), np.imag(dB)]])
        e = np.concatenate([np.real(E) / SE, np.imag(E) / SE])
        JtJ = J.T @ J
        d = np.diag(JtJ)
        aug = np.sqrt(relax * np.diag(d + np.max(d) * np.finfo(float).eps))
        A = np.vstack([J, aug])
        b = np.concatenate([e, np.zeros(nrofp)])
        dy = -np.linalg.pinv(A) @ b

        y0 = y
        y = y + dy
        Bml, Aml = theta2ba(y, n, mh, ml)
        cost = mlfdi_res(Bml, Aml, freq, X, Y, sX2, sY2, cXY, waxis)
        relerror = abs(cost - cost0) / cost0

        if cost < cost0 or GN == 1:
            y0 = y
            cost0 = cost
            iter0 = it
            relerror0 = relerror
            relax /= 2.0
        else:
            y = y0
            cost = cost0
            Bml, Aml = theta2ba(y, n, mh, ml)
            relax *= 10.0
        if verbose:
            print(f"Iter {it}: index = {iter0}, cost = {cost0:g}, rel.err = {relerror0:g}")

    Hml = ba2hm(Bml, Aml, nrofi, nrofo)
    return Hml, Hls
