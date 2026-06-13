"""Chi-squared identification test (port of ``chi2test.m``)."""

from __future__ import annotations

import numpy as np

from ._common import models_list, eval_channel


def chi2test(X, Y, freq, FRF, sCR, SYS):
    """Compare model errors against the 95% chi-squared bound.

    Parameters
    ----------
    X, Y : (nroff, nrofi)/(nroff, nrofo) spectra (used only for shape)
    freq : (nroff,) Hz
    FRF, sCR : (nroff, nrofh) measured FRF and Cramer-Rao bound
    SYS : dict ``name -> Hm`` (transfer-function arrays)

    Returns
    -------
    err : (nroff, nrofm, nrofh) squared model errors (sorted by worst model)
    var : (nroff, nrofh) chi-squared confidence bound
    tag : list[list[str]] model names per channel, worst-first
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape[0] == 1:
        X = X.T
    if Y.shape[0] == 1:
        Y = Y.T
    freq = np.asarray(freq, dtype=float).ravel()
    FRF = np.atleast_2d(FRF)
    if FRF.shape[0] != freq.size:
        FRF = FRF.T
    sCR = np.atleast_2d(sCR)
    if sCR.shape[0] != freq.size:
        sCR = sCR.T
    nrofi = X.shape[1]
    nrofo = Y.shape[1]
    nroff = freq.size
    nrofh = nrofi * nrofo

    names, models = models_list(SYS)
    nrofm = len(models)
    N_alfa = 10.5966

    var = np.zeros((nroff, nrofh))
    for h in range(nrofh):
        var[:, h] = N_alfa * sCR[:, h] / 2.0

    err = np.zeros((nroff, nrofm, nrofh))
    tag = []
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        srt = np.zeros((nroff, nrofm))
        cnt = np.zeros(nrofm)
        for m in range(nrofm):
            FRFsys = eval_channel(models[m], o, i, freq)
            e = np.abs(FRF[:, h] - FRFsys) ** 2
            err[:, m, h] = e
            mask = e > var[:, h]
            cnt[m] = np.count_nonzero(mask)
            srt[mask, m] = e[mask] - var[mask, h]
        order = np.argsort(-(np.mean(srt, axis=0) * cnt))
        err[:, :, h] = err[:, order, h]
        tag.append([names[k] for k in order])
    return err, var, tag
