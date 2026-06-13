"""Transient/offset/trend removal of periodic data (port of ``pretreat.m``)."""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend as _detrend


def pretreat(x, nrofs, fs=1.0, nroft=1, trend=0):
    """Remove transient periods, per-period offsets and (optionally) trends.

    Parameters
    ----------
    x : array_like, shape (nsamples,) or (nsamples, nchan)
    nrofs : int
        Samples per signal period.
    fs : float
        Sampling frequency [Hz].
    nroft : int
        Number of leading transient periods to discard.
    trend : int
        If > 0, also remove the per-period mean (detrend, constant).

    Returns
    -------
    y : ndarray, same column count as *x*
    time : ndarray
    """
    x = np.asarray(x, dtype=float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    nrofr, nrofc = x.shape
    nrofp = int(np.ceil(nrofr / nrofs)) - nroft

    y = np.empty((nrofs * nrofp, nrofc))
    for k in range(nrofc):
        u = x[nroft * nrofs:, k]
        u = u[:nrofs * nrofp]
        D = u.reshape(nrofs, nrofp, order="F")
        D = D - np.mean(D, axis=0, keepdims=True)
        if trend > 0:
            D = _detrend(D, axis=0, type="constant")
        y[:, k] = D.flatten(order="F")

    time = np.arange(y.shape[0]) / fs
    return (y[:, 0] if one_d else y), time
