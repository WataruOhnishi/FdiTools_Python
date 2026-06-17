"""FRF-level utilities (port of ``hfrf``/``cr_rao``/``fdicohere``/``fdel_fdi``/
``fcat_fdi``)."""

from __future__ import annotations

import numpy as np
import control

from ..frfdata import FrfData, UserData

# UserData fields that are indexed by frequency line (one row per freq).
_FREQ_FIELDS = ("X", "Y", "FRFn", "sX2", "sY2", "cXY", "sCR", "sG", "cxy")


def frfconf(p, M=None):
    """Confidence-radius factor for a measured FRF (port of ``frfconf.m``).

    The ``100*p%`` circular confidence bound on an FRF averaged from *M* periods
    has radius ``sigma_Ghat * frfconf(p, M)`` (Pintelon-Schoukens 2012 eq.2-40),
    with ``sigma_Ghat = UserData.sG``.  ``f = sqrt(F_p(2, 2M-2))``; for ``M < 2``
    the large-M limit ``sqrt(-log(1-p))`` is used.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in the open interval (0, 1).")
    if M is None or not np.isfinite(M) or M < 2:
        return np.sqrt(-np.log(1.0 - p))
    nu = 2 * M - 2
    return np.sqrt((nu / 2.0) * ((1.0 - p) ** (-2.0 / nu) - 1.0))


def hfrf(Hm, freq):
    """Evaluate a transfer-function array on *freq* (Hz).

    Returns a ``(nroff, nrofh)`` complex matrix, column ``h = (i-1)*nrofo+o``.
    """
    Hm = np.atleast_2d(np.asarray(Hm, dtype=object))
    nrofo, nrofi = Hm.shape
    freq = np.asarray(freq, dtype=float).ravel()
    nroff = freq.size
    s = 1j * 2.0 * np.pi * freq

    FRF = np.zeros((nroff, nrofi * nrofo), dtype=complex)
    for h in range(nrofi * nrofo):
        i = h // nrofo
        o = h - i * nrofo
        sys = control.tf(Hm[o, i])
        FRF[:, h] = np.array([complex(sys(sk)) for sk in s])
    return FRF


def cr_rao(X, Y, freq, B, A, sX2, sY2, cXY, n, mh, ml, cORd, fs=None):
    """Cramer-Rao lower bound of the parameter covariance (Fisher matrix).

    Port of ``cr_rao.m`` (SISO).
    """
    B = np.asarray(B, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    sX2 = np.asarray(sX2, dtype=float).ravel()
    sY2 = np.asarray(sY2, dtype=float).ravel()
    cXY = np.asarray(cXY).ravel()
    freq = np.asarray(freq, dtype=float).ravel()

    B = B[n - mh:]                       # drop the first n-mh leading entries
    y = np.concatenate((A[1:n + 1], B[:mh - ml + 1]))
    N = freq.size

    if cORd == "c":
        waxis = 1j * 2.0 * np.pi * freq
    elif cORd == "d":
        waxis = np.exp(1j * 2.0 * np.pi * freq / fs)
    else:
        waxis = 1j * 2.0 * np.pi * freq

    P = waxis[:, None] ** np.arange(n, -1, -1)[None, :]          # (N, n+1)
    Q = waxis[:, None] ** np.arange(mh, ml - 1, -1)[None, :]     # (N, mh-ml+1)

    Num = Q @ y[n:n + 1 + mh - ml]
    Den = P @ np.concatenate(([1.0], y[:n]))
    SE = np.sqrt(sX2 * np.abs(Num) ** 2 + sY2 * np.abs(Den) ** 2
                 - 2.0 * np.real(cXY * Den * np.conj(Num)))

    E = Num * X - Den * Y
    cols = []
    for i in range(1, n + 1):
        WW = (-Y * P[:, i] / SE
              - E / SE ** 3 * (sY2 * np.real(Den * np.conj(P[:, i]))
                               - np.real(cXY * P[:, i] * np.conj(Num))))
        cols.append(WW)
    for i in range(mh - ml + 1):
        WW = (X * Q[:, i] / SE
              - E / SE ** 3 * (sX2 * np.real(Num * np.conj(Q[:, i]))
                               - np.real(cXY * Den * np.conj(Q[:, i]))))
        cols.append(WW)
    Amat = np.column_stack(cols)
    J = np.vstack([np.real(Amat), np.imag(Amat)])
    return J.T @ J


def fdicohere(Pest):
    """Periodic (ensemble) coherence of a :class:`FrfData` (port of v3.0
    ``fdicohere.m``).

    Uses the period-by-period DFTs of the stored time data ``x``/``y`` at the
    excited lines ``ms.ex`` (no Signal Processing Toolbox).  For each output
    ``o`` and input ``i``::

        gamma^2 = |sum_p Y_{o,p} conj(U_{i,p})|^2
                  / ((sum_p |U_{i,p}|^2)(sum_p |Y_{o,p}|^2)).

    The result is stored in ``UserData.cxy`` with shape ``(nl, ny, nu)``.
    """
    ud = Pest.userdata
    ms = ud.ms
    nrofs = ms.nrofs
    # excited fft bins (0-based) aligned with the FRF lines; derived from the
    # FRF frequencies so it works whether or not ms carries .ex
    ex = getattr(ms, "ex", None)
    if ex is None:
        ex = np.round(np.asarray(Pest.freq) / ms.harm.df).astype(int)
    else:
        ex = np.asarray(ex, dtype=int)
    nl = ex.size

    x = np.asarray(ud.x)
    y = np.asarray(ud.y)
    U = x.reshape(x.shape[0], -1)
    Y = y.reshape(y.shape[0], -1)
    nu, ny = U.shape[1], Y.shape[1]
    M = U.shape[0] // nrofs
    if M < 2:
        raise ValueError("fdicohere: need >= 2 periods for coherence.")

    Up = np.zeros((nl, M, nu), dtype=complex)
    Yp = np.zeros((nl, M, ny), dtype=complex)
    for p in range(M):
        idx = slice(p * nrofs, (p + 1) * nrofs)
        Up[:, p, :] = np.fft.fft(U[idx, :], axis=0)[ex, :]
        Yp[:, p, :] = np.fft.fft(Y[idx, :], axis=0)[ex, :]

    cxy = np.zeros((nl, ny, nu))
    for o in range(ny):
        Yo = Yp[:, :, o]
        Syy = np.sum(np.abs(Yo) ** 2, axis=1)
        for i in range(nu):
            Ui = Up[:, :, i]
            Suu = np.sum(np.abs(Ui) ** 2, axis=1)
            Suy = np.sum(Yo * np.conj(Ui), axis=1)
            cxy[:, o, i] = np.abs(Suy) ** 2 / (Suu * Syy)
    ud.cxy = cxy
    return Pest


def _delete_freq(Pest, kmin, kmax):
    """Return a copy of *Pest* with frequency lines ``kmin..kmax`` removed."""
    keep = np.ones(Pest.freq.size, dtype=bool)
    keep[kmin:kmax + 1] = False
    ud_new = UserData()
    for name in UserData.__slots__:
        val = getattr(Pest.userdata, name, None)
        if val is None:
            setattr(ud_new, name, None)
        elif name in _FREQ_FIELDS:
            setattr(ud_new, name, np.asarray(val)[keep, ...])
        else:
            setattr(ud_new, name, val)
    return FrfData(Pest.response[:, :, keep], Pest.freq[keep], userdata=ud_new)


def fdel_fdi(sys, fmin, fmax):
    """Delete the frequency band ``[fmin, fmax]`` (Hz) from a :class:`FrfData`."""
    kmin = int(np.argmin(np.abs(sys.freq - fmin)))
    kmax = int(np.argmin(np.abs(sys.freq - fmax)))
    if kmin > kmax:
        kmin, kmax = kmax, kmin
    return _delete_freq(sys, kmin, kmax)


def fcat_fdi(*args, noise="sG"):
    """Concatenate :class:`FrfData` objects, keeping the lower-noise line on
    overlap (port of ``fcat_fdi.m``).

    Parameters
    ----------
    *args : FrfData
        Datasets to merge (must be single-channel / SISO).
    noise : str
        UserData field used to decide which overlapping line to keep.
    """
    data = list(args)
    out = data[0]
    for nxt in data[1:]:
        out = _fcat_pair(out, nxt, noise)
    return out


def _fcat_pair(sys1, sys2, noise):
    if sys1.freq.size < sys2.freq.size:
        sys1, sys2 = sys2, sys1

    # resolve overlaps line-by-line
    n1 = getattr(sys1.userdata, noise, None)
    n2 = getattr(sys2.userdata, noise, None)
    for f in sys2.freq.copy():
        idx1 = np.where(sys1.freq == f)[0]
        if idx1.size:
            k1 = idx1[0]
            k2 = np.where(sys2.freq == f)[0][0]
            v1 = np.abs(np.asarray(n1)[k1, 0]) if n1 is not None else np.inf
            v2 = np.abs(np.asarray(n2)[k2, 0]) if n2 is not None else np.inf
            if v1 < v2:
                sys2 = fdel_fdi(sys2, f, f)
                n2 = getattr(sys2.userdata, noise, None)
            else:
                sys1 = fdel_fdi(sys1, f, f)
                n1 = getattr(sys1.userdata, noise, None)

    # merge and sort by frequency
    freq = np.concatenate([sys1.freq, sys2.freq])
    order = np.argsort(freq)
    resp = np.concatenate([sys1.response, sys2.response], axis=2)[:, :, order]

    ud = UserData()
    for name in UserData.__slots__:
        v1 = getattr(sys1.userdata, name, None)
        v2 = getattr(sys2.userdata, name, None)
        if name in _FREQ_FIELDS and v1 is not None and v2 is not None:
            merged = np.concatenate([np.asarray(v1), np.asarray(v2)], axis=0)[order, ...]
            setattr(ud, name, merged)
        else:
            setattr(ud, name, v1 if v1 is not None else v2)
    return FrfData(resp, freq[order], userdata=ud)
