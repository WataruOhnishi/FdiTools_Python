"""Maximum-likelihood estimation of the FRF from periodic data
(port of ``time2frf_ml.m`` + the ``@iodata`` MIMO method).

Two calling conventions, mirroring the MATLAB function:

* structured : ``time2frf_ml(x, y, ms)`` -> :class:`FrfData`
* classical  : ``time2frf_ml(x, y, fs=.., fl=.., fh=.., df=..)`` ->
  ``(Xs, Ys, FRFs, FRFn, freq, sX2, sY2, cXY, sCR)``

SISO / SIMO use the matrix core.  **MIMO** is selected automatically when there
is more than one input (``x`` of shape ``(N, nu)``) or several experiments
(``x`` of shape ``(N, nu, ne)``): with ``ne >= nu`` an orthogonal/Hadamard
multiple-experiment solve ``G = Y/U`` is used; otherwise a single *zippered*
experiment (each input owns disjoint excited lines) is interpolated.  MIMO needs
the structured (``ms``) call and returns an ``(ny, nu, nl)`` :class:`FrfData`.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData, UserData
from ..auxiliary.frfutils import fdicohere


def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _cov_offdiag(a, b):
    """MATLAB ``cov(a,b)`` (1,2) element for complex vectors (ddof=1)."""
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * np.conj(b)) / (a.size - 1)


def _avgspec(d, nrofs, ex):
    """Period-averaged DFT of each column of *d* at bins *ex*.

    Returns ``(S, V)``: mean spectrum and per-component variance of the mean.
    """
    d = np.asarray(d, dtype=float)
    M = d.shape[0] // nrofs
    nc = d.shape[1]
    nl = ex.size
    S = np.zeros((nl, nc), dtype=complex)
    V = np.zeros((nl, nc))
    for c in range(nc):
        Pp = np.column_stack([np.fft.fft(d[p * nrofs:(p + 1) * nrofs, c])
                              for p in range(M)])
        Sp = Pp[ex, :]
        S[:, c] = Sp.mean(axis=1)
        if M > 1:
            V[:, c] = np.var(Sp, axis=1, ddof=1) / 2.0 / M
    return S, V


def _time2frf_ml_mimo(x, y, ms, df):
    """MIMO FRF from period-averaged spectra (orthogonal or zippered)."""
    fs = ms.harm.fs
    nrofs = ms.nrofs
    ex = np.asarray(ms.ex, dtype=int)
    freq = ex * df
    nl = ex.size

    if x.ndim == 2:
        x = x[:, :, None]
    if y.ndim == 2:
        y = y[:, :, None]
    nu, ne = x.shape[1], x.shape[2]
    ny = y.shape[1]
    M = x.shape[0] // nrofs

    Uall = np.zeros((nl, nu, ne), dtype=complex)
    Yall = np.zeros((nl, ny, ne), dtype=complex)
    sYa = np.zeros((nl, ny, ne))
    for e in range(ne):
        Uall[:, :, e], _ = _avgspec(x[:, :, e], nrofs, ex)
        Yall[:, :, e], sYa[:, :, e] = _avgspec(y[:, :, e], nrofs, ex)

    G = np.zeros((ny, nu, nl), dtype=complex)
    sG = np.zeros((ny, nu, nl))
    if ne >= nu:
        for f in range(nl):
            Um = Uall[f, :, :]                       # nu x ne
            Ym = Yall[f, :, :]                        # ny x ne
            Wm = np.linalg.pinv(Um)                   # ne x nu
            G[:, :, f] = Ym @ Wm
            sy = sYa[f, :, :]                         # ny x ne
            for i in range(nu):
                sG[:, i, f] = np.sqrt(sy @ np.abs(Wm[:, i]) ** 2)
        method = "orthogonal"
    else:
        Gsp = np.full((ny, nu, nl), np.nan, dtype=complex)
        Ssp = np.full((ny, nu, nl), np.nan)
        own = [[] for _ in range(nu)]
        for f in range(nl):
            uvec = Uall[f, :, 0]
            ia = int(np.argmax(np.abs(uvec)))
            Gsp[:, ia, f] = Yall[f, :, 0] / uvec[ia]
            Ssp[:, ia, f] = np.sqrt(sYa[f, :, 0]) / np.abs(uvec[ia])
            own[ia].append(f)
        for i in range(nu):
            fi = np.array(own[i], dtype=int)
            for o in range(ny):
                G[o, i, :] = np.interp(freq, freq[fi], np.real(Gsp[o, i, fi])) \
                    + 1j * np.interp(freq, freq[fi], np.imag(Gsp[o, i, fi]))
                sG[o, i, :] = np.interp(freq, freq[fi], Ssp[o, i, fi])
        method = "zippered"

    ud = UserData(ms=ms, sG=sG, nrofp=M, method=method)
    return FrfData(G, freq, userdata=ud)


def time2frf_ml(x, y, ms=None, *, fs=None, fl=None, fh=None, df=None,
                flagTime=False):
    structured = ms is not None
    if structured:
        fs, fl, fh, df = ms.harm.fs, ms.harm.fl, ms.harm.fh, ms.harm.df

    xa = np.asarray(x, dtype=float)
    nu = xa.shape[1] if xa.ndim >= 2 else 1
    ne = xa.shape[2] if xa.ndim == 3 else 1
    if structured and (nu > 1 or ne > 1):
        return _time2frf_ml_mimo(xa, np.asarray(y, dtype=float), ms, df)
    if not structured and xa.ndim == 3:
        raise ValueError("MIMO data (3-D x) requires the structured call "
                         "time2frf_ml(x, y, ms).")

    x = _as2d(x)
    y = _as2d(y)
    nrofi = x.shape[1]
    nrofo = y.shape[1]
    nrofh = nrofi * nrofo
    nrofs = int(round(fs / df))
    nl = int(np.ceil(fl / df))
    nh = int(np.floor(fh / df))
    freq = np.arange(nl, nh + 1) * df
    nroff = freq.size
    nrofp = int(np.floor(x.shape[0] / nrofs))

    # ----- signal FFT -----------------------------------------------------
    INP = np.zeros((nroff, nrofp, nrofi), dtype=complex)
    OUT = np.zeros((nroff, nrofp, nrofo), dtype=complex)
    Xs = np.zeros((nroff, nrofi), dtype=complex)
    Ys = np.zeros((nroff, nrofo), dtype=complex)
    sX2 = np.zeros((nroff, nrofi))
    sY2 = np.zeros((nroff, nrofo))

    for i in range(nrofi):
        for p in range(nrofp):
            Ip = np.fft.fft(x[p * nrofs:(p + 1) * nrofs, i])
            INP[:, p, i] = Ip[nl:nh + 1]
        Xs[:, i] = INP[:, :, i].mean(axis=1)
        sX2[:, i] = (INP[:, :, i].std(axis=1, ddof=1) ** 2) / 2.0 / nrofp
    for o in range(nrofo):
        for p in range(nrofp):
            Op = np.fft.fft(y[p * nrofs:(p + 1) * nrofs, o])
            OUT[:, p, o] = Op[nl:nh + 1]
        Ys[:, o] = OUT[:, :, o].mean(axis=1)
        sY2[:, o] = (OUT[:, :, o].std(axis=1, ddof=1) ** 2) / 2.0 / nrofp

    cXY = np.zeros((nroff, nrofh), dtype=complex)
    FRFs = np.zeros((nroff, nrofh), dtype=complex)
    sCR = np.zeros((nroff, nrofh))
    sG = np.zeros((nroff, nrofh))
    # non-excited lines carry zero input spectrum -> harmless divide-by-zero
    # (those lines are discarded for qlog excitation below)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(nrofi):
            for o in range(nrofo):
                h = i * nrofo + o
                for f in range(nroff):
                    cXY[f, h] = _cov_offdiag(INP[f, :, i], OUT[f, :, o]) / 2.0 / nrofp
                FRFs[:, h] = Ys[:, o] / Xs[:, i]
                common = (sX2[:, i] / np.abs(Xs[:, i]) ** 2
                          + sY2[:, o] / np.abs(Ys[:, o]) ** 2
                          - 2.0 * np.real(cXY[:, h] / (np.conj(Xs[:, i]) * Ys[:, o])))
                sCR[:, h] = np.sqrt(np.abs(FRFs[:, h]) ** 2 * common)
                # sG: FRF standard deviation (PS2012 eq.2-38) = sqrt(2) * sCR
                sG[:, h] = np.sqrt(2.0) * sCR[:, h]

    # ----- noise FFT (2-period blocks, in-between lines) ------------------
    nph = nrofp // 2
    Yn = np.zeros((nroff, nrofo), dtype=complex)
    FRFn = np.zeros((nroff, nrofh), dtype=complex)
    for o in range(nrofo):
        NSE = np.zeros((nroff, nph), dtype=complex)
        for p in range(nph):
            Op = np.fft.fft(y[p * nrofs * 2:(p + 1) * nrofs * 2, o])
            block = Op[2 * nl - 1:2 * nh + 1]    # length 2*nroff
            NSE[:, p] = block[::2]               # uneven (in-between) lines
        Yn[:, o] = NSE.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(nrofi):
            for o in range(nrofo):
                FRFn[:, i * nrofo + o] = Yn[:, o] / Xs[:, i]

    if not structured:
        return Xs, Ys, FRFs, FRFn, freq, sX2, sY2, cXY, sCR

    # ----- structured output: keep only excited (e.g. qlog) lines ---------
    if ms.options.gtp in ("q", "qlog"):
        freq_ex = ms.freq[np.asarray(ms.ex, dtype=int)]
        sel = np.array([int(np.argmin(np.abs(freq - fe))) for fe in freq_ex])
        freq = freq_ex
        Xs, Ys = Xs[sel], Ys[sel]
        FRFs, FRFn = FRFs[sel], FRFn[sel]
        sX2, sY2, cXY = sX2[sel], sY2[sel], cXY[sel]
        sCR, sG = sCR[sel], sG[sel]

    ud = UserData(X=Xs, Y=Ys, FRFn=FRFn, sX2=sX2, sY2=sY2, cXY=cXY,
                  sCR=sCR, sG=sG, nrofp=nrofp, ms=ms)
    Pest = FrfData(FRFs, freq, nrofi=nrofi, nrofo=nrofo, userdata=ud)

    if flagTime:
        ud.x = x
        ud.y = y
        fdicohere(Pest)
    return Pest
