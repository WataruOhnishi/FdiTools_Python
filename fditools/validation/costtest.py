"""Maximum-likelihood cost validation test (port of ``costtest.m``)."""

from __future__ import annotations

import numpy as np

from ._common import models_list, denom_order
from ..auxiliary.conversions import tfdata
from ..parametric._residuals import fdicost


def costtest(X, Y, freq, sX2, sY2, cXY, SYS, relax, nrofp):
    """Return ``(cost, interval, tag)``.

    cost : (nrofm, nrofh) residual cost per model/channel (sorted worst-first)
    interval : (lo, hi) expected-cost confidence interval
    tag : list[list[str]] model names, worst-first
    """
    def _col(a):
        a = np.atleast_2d(a)
        return a if a.shape[0] == np.asarray(freq).size else a.T

    freq = np.asarray(freq, dtype=float).ravel()
    X, Y = _col(X), _col(Y)
    sX2, sY2, cXY = _col(sX2), _col(sY2), _col(cXY)
    nrofi = X.shape[1]
    nrofo = Y.shape[1]
    nrofh = nrofi * nrofo

    names, models = models_list(SYS)
    nrofm = len(models)

    n = denom_order(models[0])
    Vnoise = ((nrofp - 1) / (nrofp - 2)) * (freq.size - n)
    interval = (Vnoise - 2.0 * np.sqrt(Vnoise), Vnoise + 2.0 * np.sqrt(Vnoise))

    cost = np.zeros((nrofm, nrofh))
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        for m in range(nrofm):
            num, den = tfdata(models[m][o, i])
            Bn = np.zeros((1, den.size))
            Bn[0, den.size - num.size:] = num          # left-pad to denom length
            cost[m, h] = fdicost(Bn, den, freq, X[:, i], Y[:, o],
                                 sX2[:, i], sY2[:, o], cXY[:, h], relax)
    order = np.argsort(-np.sum(cost, axis=1))
    cost = cost[order, :]
    tag = [names[k] for k in order]
    return cost, interval, tag
