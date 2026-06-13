"""Best Linear Approximation of the FRF (port of ``time2bla.m``).

``x``/``y`` hold several realisations (measurements) column-by-column.
"""

from __future__ import annotations

import numpy as np


def time2bla(x, y, fs, fl, fh, df):
    """Return ``(X, Y, FRF, freq, Gbla, sX2, sY2, cXY)``.

    Gbla rows: ``[mean FRF, std total, std noise, std nonlinear]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    if x.shape[0] < x.shape[1]:
        x = x.T
        y = y.T

    nrofs = int(round(fs / df))
    nl = int(np.ceil(fl / df))
    nh = int(np.floor(fh / df))
    freq = np.arange(nl, nh + 1) * df
    nroff = freq.size
    nrofp = x.shape[0] // nrofs
    nrofm = x.shape[1]

    Xf = np.zeros((nrofm, nrofp, nroff), dtype=complex)
    Yf = np.zeros((nrofm, nrofp, nroff), dtype=complex)
    Ff = np.zeros((nrofm, nrofp, nroff), dtype=complex)
    for m in range(nrofm):
        for p in range(nrofp):
            Xp = np.fft.fft(x[p * nrofs:(p + 1) * nrofs, m])
            Yp = np.fft.fft(y[p * nrofs:(p + 1) * nrofs, m])
            Xf[m, p, :] = Xp[nl:nh + 1]
            Yf[m, p, :] = Yp[nl:nh + 1]
            Ff[m, p, :] = Yf[m, p, :] / Xf[m, p, :]

    Gbla = np.zeros((4, nroff), dtype=complex)
    mean_over_p = Ff.mean(axis=1)                              # (nrofm, nroff)
    Gbla[0, :] = mean_over_p.mean(axis=0)                      # overall mean
    Gbla[1, :] = mean_over_p.std(axis=0, ddof=1) / np.sqrt(nrofm)
    std_over_p = Ff.std(axis=1, ddof=1)                        # (nrofm, nroff)
    Gbla[2, :] = (np.mean(std_over_p ** 2, axis=0) / (nrofm * nrofp)) ** 0.5
    Gbla[3, :] = (nrofm * np.abs(Gbla[1, :] ** 2 - Gbla[2, :] ** 2)) ** 0.5

    sXp = np.zeros((nrofm, nroff))
    sYp = np.zeros((nrofm, nroff))
    cXYp = np.zeros((nrofm, nroff), dtype=complex)
    for m in range(nrofm):
        sXp[m, :] = (Xf[m].std(axis=0, ddof=1) ** 2) / 2.0 / nrofp
        sYp[m, :] = (Yf[m].std(axis=0, ddof=1) ** 2) / 2.0 / nrofp
        for ii in range(nroff):
            a = Xf[m, :, ii] - Xf[m, :, ii].mean()
            b = Yf[m, :, ii] - Yf[m, :, ii].mean()
            cXYp[m, ii] = np.sum(a * np.conj(b)) / (nrofp - 1) / 2.0 / nrofp
    sX2 = sXp.mean(axis=0)
    sY2 = sYp.mean(axis=0)
    cXY = cXYp.mean(axis=0)

    X = Xf.mean(axis=1).mean(axis=0)
    Y = Yf.mean(axis=1).mean(axis=0)
    FRF = Y / X
    sCR = 2.0 * np.abs(FRF) * (sX2 / np.abs(X) ** 2
                              + sY2 / np.abs(Y) ** 2
                              - 2.0 * np.real(cXY / (np.conj(X) * Y)))
    return X, Y, FRF, freq, Gbla, sX2, sY2, cXY
