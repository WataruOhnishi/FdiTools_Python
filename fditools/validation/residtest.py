"""Whiteness-of-residuals identification test (port of ``residtest.m``)."""

from __future__ import annotations

import numpy as np

from ._common import models_list, eval_channel


def _xcorr_unbiased(x):
    """Unbiased auto-correlation, full length ``2N-1`` (MATLAB ``xcorr(.,'unbiased')``)."""
    x = np.asarray(x)
    N = x.size
    full = np.correlate(x, x, mode="full")        # lags -N+1 .. N-1
    scale = N - np.abs(np.arange(-N + 1, N))
    return full / scale


def residtest(x, y, freq, FRF, SYS, sCR, fs):
    """Return ``(lags, corr, cb50, frac50, tag, cb95, frac95)``."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.shape[0] == 1:
        x = x.T
    if y.shape[0] == 1:
        y = y.T
    freq = np.asarray(freq, dtype=float).ravel()
    FRF = np.atleast_2d(FRF)
    if FRF.shape[0] != freq.size:
        FRF = FRF.T
    sCR = np.atleast_2d(sCR)
    if sCR.shape[0] != freq.size:
        sCR = sCR.T

    nroff = freq.size
    nrofi = x.shape[1]
    nrofo = y.shape[1]
    nrofh = nrofi * nrofo
    names, models = models_list(SYS)
    nrofm = len(models)

    lags = np.arange(-nroff + 1, nroff)            # length 2*nroff-1
    nrofp = x.shape[0] / fs * (freq[1] - freq[0])

    scale0 = (nrofp - 2) / (nrofp - 1)
    scale = (nrofp - 5.0 / 3.0) / (nrofp - 11.0 / 12.0)
    cb_scale0 = scale0 * ((nrofp - 1) ** 1.5 / (nrofp - 2) / (nrofp - 3) ** 0.5)
    cb_scale = scale * (nrofp - 1) / (nrofp - 2) * np.ones_like(lags, dtype=float)
    cb_scale[nroff - 1] = cb_scale0
    ac_scale = scale * np.ones_like(lags, dtype=float)
    ac_scale[nroff - 1] = scale0

    p50 = np.sqrt(-np.log(1 - 0.5))
    p95 = np.sqrt(-np.log(1 - 0.95))
    conf = cb_scale / (nroff - np.abs(lags)) ** 0.5
    cb50 = p50 * conf
    cb95 = p95 * conf

    select = np.ones(2 * nroff - 1, dtype=bool)
    select[nroff - 1] = False                      # drop zero lag

    frac50 = np.zeros((nrofm, nrofh))
    frac95 = np.zeros((nrofm, nrofh))
    corr = np.zeros((lags.size, nrofm, nrofh))
    tag = []
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        for m in range(nrofm):
            FRFsys = eval_channel(models[m], o, i, freq)
            res = (FRF[:, h] - FRFsys) / sCR[:, h] ** 0.5
            ac = _xcorr_unbiased(res) * ac_scale
            frac50[m, h] = np.count_nonzero(
                np.abs(ac[select]) - cb50[select] > 0) / (2 * nroff - 2) * 100
            frac95[m, h] = np.count_nonzero(
                np.abs(ac[select]) - cb95[select] > 0) / (2 * nroff - 2) * 100
            corr[:, m, h] = np.abs(ac)
        order = np.argsort(-frac50[:, h])
        frac50[:, h] = frac50[order, h]
        frac95[:, h] = frac95[order, h]
        corr[:, :, h] = corr[:, order, h]
        tag.append([names[k] for k in order])
    return lags, corr, cb50, frac50, tag, cb95, frac95
