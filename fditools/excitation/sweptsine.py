"""Swept-sine (chirp) excitation generation (port of ``sweptsine.m``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _ns(obj):
    return SimpleNamespace(**obj) if isinstance(obj, dict) else obj


def sweptsine(h, options):
    """Generate a linear/quadratic/logarithmic swept sine.

    Parameters
    ----------
    h : dict | namespace with ``fs``, ``fl``, ``fh``, ``df``
    options : dict | namespace with ``type`` in {'lin','qdr','log'}

    Returns
    -------
    SimpleNamespace with ``x``, ``time``, ``X``, ``freq``.
    """
    h = _ns(h)
    options = _ns(options)
    nrofs = int(np.ceil(h.fs / h.df))
    time = np.arange(nrofs) / h.fs
    freq = h.fs * np.arange(nrofs // 2) / nrofs

    t = options.type
    if t in ("lin", "linear"):
        x = np.sin(np.pi * (h.fh - h.fl) * h.df * time ** 2 + 2 * np.pi * h.fl * time)
    elif t in ("qdr", "quadratic"):
        x = np.sin(2.0 / 3.0 * np.pi * (h.fh - h.fl) * h.df ** 2 * time ** 3
                   + 2 * np.pi * h.fl * time)
    elif t in ("log", "logarithmic"):
        x = np.sin(2 * np.pi / h.df / np.log(h.fh / h.fl)
                   * (h.fl * (h.fh / h.fl) ** (time * h.df) - h.fl))
    else:
        raise ValueError(f"unknown sweep type {t!r}")

    X = np.fft.fft(x, nrofs)
    half = int(np.floor(nrofs / 2))
    X = np.concatenate([X[0:1], 2.0 * X[1:half]]) / nrofs
    return SimpleNamespace(x=x, time=time, X=X, freq=freq)
