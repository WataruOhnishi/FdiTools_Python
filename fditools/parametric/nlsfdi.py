"""Non-linear least-squares FDI (Levenberg-Marquardt) — port of ``nlsfdi.m``.

Structured : ``nlsfdi(Pest, FRF_W, n, mh, ml, max_iter, max_err, GN, cORd)``
Classical  : ``nlsfdi(FRF, freq, FRF_W, n, mh, ml, max_iter, max_err, GN, cORd, fs)``
Returns ``(Hnls, Hwls)``.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData
from ..aux.conversions import theta2ba, ba2theta, ba2hm, hm2ba, vectorize_orders
from .wlsfdi import wlsfdi
from ._residuals import nlsfdi_res


def _coerce(a, nroff, nrofh, dtype):
    a = np.atleast_2d(np.asarray(a, dtype=dtype))
    if a.shape[0] != nroff:
        a = a.T
    if a.shape[1] != nrofh:
        a = np.tile(a.reshape(nroff, -1), (1, nrofh // a.shape[1]))
    return a


def nlsfdi(*args, verbose=False):
    if isinstance(args[0], FrfData):
        Pest, FRF_W, n, M_mh, M_ml, max_iter, max_err, GN, cORd = args[:9]
        FRF = Pest.frf_columns()
        freq = Pest.freq
        ms = Pest.userdata.ms
        fs = (ms[0] if isinstance(ms, (list, tuple)) else ms).harm.fs
    else:
        FRF, freq, FRF_W, n, M_mh, M_ml, max_iter, max_err, GN, cORd, fs = args[:11]

    freq = np.asarray(freq, dtype=float).ravel()
    nroff = freq.size
    mh_mat = np.atleast_2d(np.asarray(M_mh))
    nrofo, nrofi = mh_mat.shape
    nrofh = nrofi * nrofo
    mh = vectorize_orders(M_mh)
    ml = vectorize_orders(M_ml)
    nrofb = int(np.sum(mh - ml)) + nrofh
    nrofp = nrofb + n

    FRF = _coerce(FRF, nroff, nrofh, complex)
    FRF_W = _coerce(FRF_W, nroff, nrofh, float)

    Hwls, waxis = wlsfdi(FRF, freq, FRF_W, n, M_mh, M_ml, cORd, fs)

    relax = 0.0 if GN == 1 else 0.01
    direction = 1
    Bn, An = hm2ba(Hwls)
    iter0 = 0
    relerror0 = np.inf
    relerror = np.inf
    y = ba2theta(Bn, An, n, mh, ml)
    P_fr = waxis[:, None] ** np.arange(n, -1, -1)        # (nroff, n+1)
    cost0 = nlsfdi_res(Bn, An, freq, FRF, FRF_W, cORd, fs)

    it = 0
    while it < max_iter and abs(relerror) > max_err:
        it += 1
        Ajw = P_fr @ An                                  # (nroff,)
        Bjw = np.column_stack([P_fr @ Bn[h, :] for h in range(nrofh)])  # (nroff,nrofh)

        E = []
        for h in range(nrofh):
            E.append((FRF[:, h] - Bjw[:, h] / Ajw) * FRF_W[:, h])
        E = np.concatenate(E)

        WD2 = FRF_W / (Ajw[:, None] ** 2)                # FRF_W / Ajw^2
        WD1 = FRF_W / Ajw[:, None]                        # FRF_W / Ajw
        dA = np.zeros((nrofh * nroff, n), dtype=complex)
        dB = np.zeros((nrofh * nroff, nrofb), dtype=complex)
        index = 0
        for h in range(nrofh):
            rows = slice(h * nroff, (h + 1) * nroff)
            dA[rows, :] = (waxis[:, None] ** np.arange(n - 1, -1, -1)) \
                * (WD2[:, h] * Bjw[:, h])[:, None]
            cnt = mh[h] - ml[h] + 1
            U = (waxis[:, None] ** np.arange(mh[h], ml[h] - 1, -1)) \
                * (-WD1[:, h])[:, None]
            dB[rows, index:index + cnt] = U
            index += cnt

        J = np.block([[np.real(dA), np.real(dB)],
                      [np.imag(dA), np.imag(dB)]])
        e = np.concatenate([np.real(E), np.imag(E)])
        JtJ = J.T @ J
        d = np.diag(JtJ)
        aug = np.sqrt(relax * np.diag(d + np.max(d) * np.finfo(float).eps))
        A = np.vstack([J, aug])
        b = np.concatenate([e, np.zeros(nrofp)])
        dy = -np.linalg.pinv(A) @ b

        y0 = y
        y = y + dy * direction
        Bn, An = theta2ba(y, n, mh, ml)
        cost = nlsfdi_res(Bn, An, freq, FRF, FRF_W, cORd, fs)
        relerror = (cost - cost0) / cost0

        if cost < cost0 or GN == 1:
            y0 = y
            cost0 = cost
            iter0 = it
            relerror0 = relerror
            relax /= 2.0
        else:
            y = y0
            cost = cost0
            Bn, An = theta2ba(y, n, mh, ml)
            relax *= 10.0
        if relax > 1e5:
            direction = -direction
            relax = 0.1
        if verbose:
            print(f"Iter {it}: cost = {cost0:g}, err = {relerror0:g}, r = {relax:g}")

    Hnls = ba2hm(Bn, An, nrofi, nrofo)
    return Hnls, Hwls
