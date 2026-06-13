"""Least-squares B-spline fitting (drift/trend removal).

The original ``splinefit.m`` (J. Lundgren) is a large, feature-rich least-
squares spline fitter (arbitrary breaks, periodic/robust fits, derivative
constraints).  Here we expose the practically used capability — a least-squares
spline of a chosen order through given (or uniformly spaced) breakpoints — on
top of SciPy.  Robustness iterations and derivative constraints are **not**
ported; for those keep using the MATLAB version.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def splinefit(x, y, breaks, order=4, periodic=False):
    """Fit a least-squares spline.

    Parameters
    ----------
    x, y : array_like
        Data points.
    breaks : int or array_like
        Number of (uniform) breakpoints, or an explicit breakpoint vector
        spanning ``[min(x), max(x)]``.
    order : int
        Spline order (4 = cubic, i.e. degree 3).
    periodic : bool
        Currently unsupported (raises if set ``True``).

    Returns
    -------
    spline : scipy.interpolate.LSQUnivariateSpline
        Callable; evaluate with ``spline(xx)``.
    """
    if periodic:
        raise NotImplementedError("splinefit: periodic fit not ported")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order_sorted = np.argsort(x)
    x, y = x[order_sorted], y[order_sorted]

    if np.isscalar(breaks):
        brk = np.linspace(x[0], x[-1], int(breaks))
    else:
        brk = np.asarray(breaks, dtype=float)

    interior = brk[1:-1]
    k = order - 1
    return LSQUnivariateSpline(x, y, t=interior, k=k)
