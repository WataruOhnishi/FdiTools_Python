"""Shared helpers for the validation tests."""

from __future__ import annotations

import numpy as np
import control

from ..auxiliary.conversions import tfdata


def models_list(SYS):
    """Return ``(names, models)`` from a dict of ``name -> Hm`` (object array)."""
    names = list(SYS.keys())
    models = [np.atleast_2d(np.asarray(SYS[k], dtype=object)) for k in names]
    return names, models


def eval_channel(Hm, o, i, freq):
    """Evaluate channel ``(o, i)`` of a transfer-function array on *freq* (Hz)."""
    sys = control.tf(Hm[o, i])
    s = 1j * 2.0 * np.pi * np.asarray(freq, dtype=float).ravel()
    return np.array([complex(sys(sk)) for sk in s])


def denom_order(Hm):
    """Denominator order of channel (0,0)."""
    _, den = tfdata(Hm[0, 0])
    return den.size - 1
