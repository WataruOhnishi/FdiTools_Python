"""Best Linear Approximation of the FRF (port of ``time2bla.m`` and the
``@iodata`` MIMO ``time2bla``).

* :func:`time2bla` - SISO matrix core (several realisations column-by-column).
* :func:`time2bla_mimo` - robust MIMO BLA from ``M`` random-phase realisations,
  separating the noise level from the stochastic non-linear distortion level.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData, UserData


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


def time2bla_mimo(x, y, ms, M):
    """Robust MIMO Best Linear Approximation (port of ``@iodata/time2bla``).

    Parameters
    ----------
    x, y : (N, nch, ne) arrays
        ``ne = M * nu`` experiments ordered **realization-major** (realization 1
        with its ``nu`` orthogonal experiments, then realization 2, ...).
        Each experiment is periodic with >= 2 periods; different realizations
        use different random phases.
    ms : multisine (carries the excited lines ``ex``)
    M : int   number of realizations

    Returns
    -------
    dict with ``G`` (:class:`FrfData`, the BLA), ``freq``, ``sG_total``,
    ``sG_noise``, ``sG_nl`` (= sqrt(max(total^2 - noise^2, 0))), ``M``, ``nrofp``.
    """
    from ..nonparametric.time2frf_ml import time2frf_ml

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ne = x.shape[2]
    if ne % M != 0:
        raise ValueError(f"ne ({ne}) must be a multiple of M ({M}).")
    nper = ne // M                                    # experiments per realization

    Gall = Vn = None
    freq = None
    nrofp = None
    for m in range(M):
        idx = slice(m * nper, (m + 1) * nper)
        Pm = time2frf_ml(x[:, :, idx], y[:, :, idx], ms)
        if m == 0:
            ny, nu, nl = Pm.response.shape
            Gall = np.zeros((ny, nu, nl, M), dtype=complex)
            Vn = np.zeros((ny, nu, nl, M))
            freq = Pm.freq
            nrofp = Pm.userdata.nrofp
        Gall[:, :, :, m] = Pm.response
        Vn[:, :, :, m] = np.asarray(Pm.userdata.sG) ** 2

    Gbla = Gall.mean(axis=3)
    Vtot = np.var(Gall, axis=3, ddof=1) / M           # var of the mean (noise+NL)
    Vnoise = Vn.mean(axis=3) / M
    Vnl = np.maximum(Vtot - Vnoise, 0.0)

    ud = UserData(sG=np.sqrt(Vtot), method="bla", nrofp=nrofp)
    return {"G": FrfData(Gbla, freq, userdata=ud), "freq": freq,
            "sG_total": np.sqrt(Vtot), "sG_noise": np.sqrt(Vnoise),
            "sG_nl": np.sqrt(Vnl), "M": M, "nrofp": nrofp}
