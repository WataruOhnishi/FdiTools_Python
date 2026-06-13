"""Conversions between parameter vectors, polynomial matrices and transfer
function arrays (port of ``ba2theta``/``theta2ba``/``ba2hm``/``hm2ba``).

Transfer-function arrays ``Hm`` are represented as a 2-D ``object`` ndarray of
SISO :class:`control.TransferFunction`, indexed ``Hm[o, i]`` (output, input) to
mirror the MATLAB ``Hm(o, i)`` convention.
"""

from __future__ import annotations

import numpy as np
import control


def vectorize_orders(orders):
    """Replicate MATLAB ``M = M'; M = M(:)`` for numerator-order matrices.

    A flat sequence ``[m1, m2, ...]`` is interpreted as one entry per transfer
    function (SISO/SIMO/MISO).  A 2-D ``(nrofo, nrofi)`` array reproduces the
    column-major ordering of the original code.
    """
    arr = np.atleast_2d(np.asarray(orders))
    return arr.T.flatten(order="F").astype(int)


def tfdata(H):
    """Return ``(num, den)`` 1-D coefficient arrays of a SISO transfer function."""
    H = control.tf(H)
    num = np.atleast_1d(np.asarray(H.num[0][0], dtype=float).ravel())
    den = np.atleast_1d(np.asarray(H.den[0][0], dtype=float).ravel())
    return num, den


def ba2theta(Bn, An, n, M_mh, M_ml):
    """Rational polynomial matrices -> estimation parameter vector ``theta``.

    Bn : (nrofh, n+1) numerator coefficient matrix (highest power first)
    An : (n+1,) common denominator coefficients (monic, ``An[0] == 1``)
    """
    Bn = np.atleast_2d(np.asarray(Bn, dtype=float))
    An = np.asarray(An, dtype=float).ravel()
    M_mh = vectorize_orders(M_mh)
    M_ml = vectorize_orders(M_ml)
    nrofh = M_ml.size

    nrofb = int(np.sum(M_mh - M_ml)) + nrofh
    y = np.zeros(n + nrofb)
    y[:n] = An[1:n + 1]

    index = n
    for i in range(nrofh):
        lo = n - M_mh[i]
        hi = n - M_ml[i]  # inclusive
        yy = Bn[i, lo:hi + 1]
        cnt = M_mh[i] - M_ml[i] + 1
        y[index:index + cnt] = yy
        index += cnt
    return y


def theta2ba(y, n, M_mh, M_ml):
    """Estimation parameter vector ``theta`` -> rational polynomial matrices.

    Returns ``(Bn, An)`` with ``Bn`` shape ``(nrofh, n+1)`` and ``An`` shape
    ``(n+1,)`` (monic).
    """
    # GTLS/BTLS produce complex parameter vectors whose imaginary parts are
    # numerical noise; keep the real part (matches the real MATLAB result).
    y = np.real(np.asarray(y)).astype(float).ravel()
    M_mh = vectorize_orders(M_mh)
    M_ml = vectorize_orders(M_ml)
    nrofh = M_ml.size

    An = np.concatenate(([1.0], y[:n]))
    Bn = np.zeros((nrofh, n + 1))
    index = n
    for i in range(nrofh):
        cnt = M_mh[i] - M_ml[i] + 1
        yy = y[index:index + cnt]
        lo = n - M_mh[i]
        hi = n - M_ml[i]  # inclusive
        Bn[i, lo:hi + 1] = yy
        index += cnt
    return Bn, An


def ba2hm(Bn, An, nrofi, nrofo):
    """Polynomial matrices -> ``(nrofo, nrofi)`` object array of SISO ``tf``."""
    Bn = np.atleast_2d(np.asarray(Bn, dtype=float))
    An = np.asarray(An, dtype=float).ravel()
    Hm = np.empty((nrofo, nrofi), dtype=object)
    for h in range(nrofi * nrofo):
        i = h // nrofo
        o = h - i * nrofo
        Hm[o, i] = control.tf(Bn[h, :], An)
    return Hm


def hm2ba(Hm):
    """``(nrofo, nrofi)`` object array of SISO ``tf`` -> ``(Bn, An)``.

    A common denominator is assumed (taken from the first channel); every
    numerator is left-padded to the denominator length.
    """
    Hm = np.atleast_2d(np.asarray(Hm, dtype=object))
    nrofo, nrofi = Hm.shape
    nrofh = nrofi * nrofo

    _, An = tfdata(Hm[0, 0])
    na = An.size
    Bn = np.zeros((nrofh, na))
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        num, den = tfdata(Hm[o, i])
        Bn[h, na - num.size:] = num
        if h == 0:
            An = den
    return Bn, An
