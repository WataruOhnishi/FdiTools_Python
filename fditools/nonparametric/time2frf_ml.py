"""Maximum-likelihood estimation of the FRF from periodic data
(port of ``time2frf_ml.m``; SISO / SIMO).

Two calling conventions, mirroring the MATLAB function:

* structured : ``time2frf_ml(x, y, ms)`` -> :class:`FrfData`
* classical  : ``time2frf_ml(x, y, fs=.., fl=.., fh=.., df=..)`` ->
  ``(Xs, Ys, FRFs, FRFn, freq, sX2, sY2, cXY, sCR)``
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


def time2frf_ml(x, y, ms=None, *, fs=None, fl=None, fh=None, df=None,
                flagTime=False):
    structured = ms is not None
    if structured:
        fs, fl, fh, df = ms.harm.fs, ms.harm.fl, ms.harm.fh, ms.harm.df

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
    sGhat = np.zeros((nroff, nrofh))
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
                # sGhat: upstream fix "correct sGhat"
                # (HoriFujimotoLab/FdiTools @3307555) -> sqrt(2) * sCR
                sGhat[:, h] = np.sqrt(2.0) * sCR[:, h]

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
        sCR, sGhat = sCR[sel], sGhat[sel]

    ud = UserData(X=Xs, Y=Ys, FRFn=FRFn, sX2=sX2, sY2=sY2, cXY=cXY,
                  sCR=sCR, sGhat=sGhat, ms=ms)
    Pest = FrfData(FRFs, freq, nrofi=nrofi, nrofo=nrofo, userdata=ud)

    if flagTime:
        ud.x = x
        ud.y = y
        fdicohere(Pest)
    return Pest
