"""Bootstrapped total least-squares FDI (port of ``btlsfdi.m``).

Structured : ``btlsfdi(Pest, n, mh, ml, relax, max_iter, max_err, cORd)``
Classical  : ``btlsfdi(X, Y, freq, n, mh, ml, sY2, sX2, cXY, relax, max_iter,
              max_err, cORd, fs)``
Returns ``(Hbtls, Hgtls)``.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData
from ..auxiliary.conversions import theta2ba, ba2theta, ba2hm, hm2ba, vectorize_orders
from ._residuals import waxis_of, qsvd, btlsfdi_res
from .gtlsfdi import gtlsfdi, _chol_block, _solve_qsvd, _as2d


def btlsfdi(*args, verbose=False):
    if isinstance(args[0], FrfData):
        Pest, n, M_mh, M_ml, relax, max_iter, max_err, cORd = args[:8]
        ud = Pest.userdata
        X, Y = ud.X, ud.Y
        freq = Pest.freq
        sX2, sY2, cXY = ud.sX2, ud.sY2, ud.cXY
        ms = ud.ms
        fs = (ms[0] if isinstance(ms, (list, tuple)) else ms).harm.fs
    else:
        (X, Y, freq, n, M_mh, M_ml, sY2, sX2, cXY,
         relax, max_iter, max_err, cORd, fs) = args[:14]

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

    Hgtls, waxis = gtlsfdi(X, Y, freq, n, mh, ml, sX2, sY2, cXY, cORd, fs)

    Bb, Ab = hm2ba(Hgtls)
    Xg0 = np.concatenate(([1.0], ba2theta(Bb, Ab, n, mh, ml)))
    err = max_err + 1
    it = 0
    while it <= max_iter and err > max_err:
        it += 1
        Den = np.polyval(Ab, waxis)
        Num = np.column_stack([np.polyval(Bb[h, :], waxis) for h in range(nrofh)])

        # weighted Jacobian
        P = np.zeros((nrofh * nroff, n + 1), dtype=complex)
        Q = np.zeros((nrofh * nroff, nrofb), dtype=complex)
        index = 0
        for h in range(nrofh):
            i = h // nrofo
            o = h - i * nrofo
            rows = slice(h * nroff, (h + 1) * nroff)
            SEr = np.sqrt((sX2[:, i] * np.abs(Num[:, h]) ** 2
                           + sY2[:, o] * np.abs(Den) ** 2
                           - 2.0 * np.real(cXY[:, h] * Den * np.conj(Num[:, h]))) ** relax)
            Xh = X[:, i] / SEr
            Yh = Y[:, o] / SEr
            P[rows, :] = (waxis[:, None] ** np.arange(n, -1, -1)) * Yh[:, None]
            cnt = mh[h] - ml[h] + 1
            U = (waxis[:, None] ** np.arange(mh[h], ml[h] - 1, -1)) * Xh[:, None]
            Q[rows, index:index + cnt] = U
            index += cnt
        J = np.block([[np.real(P), -np.real(Q)], [np.imag(P), -np.imag(Q)]])

        C = np.zeros((nrofh * (nrofp + 1), nrofp + 1), dtype=complex)
        for h in range(nrofh):
            i = h // nrofo
            o = h - i * nrofo
            cnt = mh[h] - ml[h] + 1
            SEr2 = (sX2[:, i] * np.abs(Num[:, h]) ** 2
                    + sY2[:, o] * np.abs(Den) ** 2
                    - 2.0 * np.real(cXY[:, h] * Den * np.conj(Num[:, h]))) ** relax
            MytMy = np.zeros((n + 1, n + 1), dtype=complex)
            MxtMx = np.zeros((cnt, cnt), dtype=complex)
            MytMx = np.zeros((n + 1, cnt), dtype=complex)
            for p in range(n, -1, -1):
                for q in range(n, -1, -1):
                    MytMy[n - p, n - q] = 2.0 * np.real(((-1) ** q)
                        * np.sum(waxis ** (p + q) * (sY2[:, o] / SEr2)))
            for p in range(mh[h], ml[h] - 1, -1):
                for q in range(mh[h], ml[h] - 1, -1):
                    MxtMx[mh[h] - p, mh[h] - q] = 2.0 * np.real(((-1) ** q)
                        * np.sum(waxis ** (p + q) * (sX2[:, i] / SEr2)))
            for p in range(n, -1, -1):
                for q in range(mh[h], ml[h] - 1, -1):
                    MytMx[n - p, mh[h] - q] = -2.0 * np.real(((-1) ** q)
                        * np.sum(waxis ** (p + q) * (cXY[:, h] / SEr2)))
            block = _chol_block(MytMy, MytMx, MxtMx)
            r0 = h * (nrofp + 1)
            C[r0:r0 + (n + 1 + cnt), :(n + 1 + cnt)] = block

        Xg = _solve_qsvd(J, C, nrofp)
        Bb, Ab = theta2ba(Xg[1:], n, mh, ml)
        err = np.max(np.abs((Xg0 - Xg) / Xg))
        Xg0 = Xg
        cost = btlsfdi_res(Bb, Ab, freq, X, Y, sX2, sY2, cXY, relax, waxis)
        if verbose:
            print(f"Iter {it}: cost = {cost:g}, rel.err = {err:g}")

    Hbtls = ba2hm(Bb, Ab, nrofi, nrofo)
    return Hbtls, Hgtls
