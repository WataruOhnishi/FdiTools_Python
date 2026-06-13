"""Loaders for the original MATLAB measurement ``.mat`` files.

``MultisineTypeA.mat`` / ``MultisineTypeB.mat`` are classic v5 MAT-files whose
numeric arrays and parameter structs are read directly with SciPy — no MATLAB
needed.  The benchmark model ``20160829_ident.mat`` instead stores MATLAB
*control objects* (``ss``/``tf``) that SciPy cannot reconstruct; convert it once
with ``Examples/private/convert_ident_to_python.m`` (see :func:`load_ident`).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

_PRIV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "Examples", "private")


def _matpath(name):
    return os.path.join(_PRIV, name)


def load_typeA():
    """Schroeder-multisine SIMO experiment (motor bench, 1 input / 2 outputs).

    Returns
    -------
    ms : SimpleNamespace
        Minimal multisine descriptor (``nrofs``, ``harm``, ``options``) –
        enough for ``time2frf_ml``/``pretreat`` (the original ``ms.x`` spectrum
        is not needed here).
    u : ndarray, shape (N,)        input current ``iq_adx``
    y : ndarray, shape (N, 2)      outputs ``[theta_mx, -theta_my]``
    """
    d = sio.loadmat(_matpath("MultisineTypeA.mat"),
                    struct_as_record=False, squeeze_me=True)
    msm = d["ms"]
    harm = SimpleNamespace(fs=float(msm.harm.fs), df=float(msm.harm.df),
                           fl=float(msm.harm.fl), fh=float(msm.harm.fh),
                           fr=float(getattr(msm.harm, "fr", 1.02)))
    options = SimpleNamespace(itp=str(msm.options.itp), ctp=str(msm.options.ctp),
                              gtp=str(msm.options.gtp),
                              dtp=str(getattr(msm.options, "dtp", "f")))
    ms = SimpleNamespace(nrofs=int(msm.nrofs), harm=harm, options=options)

    u = np.asarray(d["iq_adx"], dtype=float).ravel()
    y = np.column_stack([np.asarray(d["theta_mx"], dtype=float).ravel(),
                         -np.asarray(d["theta_my"], dtype=float).ravel()])
    return ms, u, y


def load_typeB():
    """Random odd-odd multisine experiment (for non-linear distortion analysis).

    Returns ``(params, u, y)`` where *params* carries ``fs, df, fl, fh, nrofs``.
    """
    d = sio.loadmat(_matpath("MultisineTypeB.mat"),
                    struct_as_record=False, squeeze_me=True)
    params = SimpleNamespace(fs=float(d["fs"]), df=float(d["df"]),
                             fl=float(d["fl"]), fh=float(d["fh"]),
                             nrofs=int(d["nrofs"]))
    u = np.asarray(d["iq_adx"], dtype=float).ravel()
    y = np.column_stack([np.asarray(d["theta_mx"], dtype=float).ravel(),
                         -np.asarray(d["theta_my"], dtype=float).ravel()])
    return params, u, y


def load_ident(channel="Pv", which=(0, 0)):
    """Load the benchmark plant, converted from ``20160829_ident.mat``.

    Run ``Examples/private/convert_ident_to_python.m`` in MATLAB first; it writes
    ``Examples/private/ident_python.mat`` with the state-space data.  Returns a
    ``control.StateSpace`` for the requested entry (default ``Pv(1,1)``).
    """
    import control

    path = _matpath("ident_python.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "ident_python.mat not found. Convert the benchmark model once in "
            "MATLAB:\n    >> cd Examples/private\n    >> convert_ident_to_python\n"
            "(script provided at Examples/private/convert_ident_to_python.m)")
    d = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    models = d["models"]
    o, i = which
    key = f"{channel}_{o + 1}{i + 1}"          # e.g. 'Pv_11'
    g = getattr(models, key)
    A = np.atleast_2d(g.A)
    B = np.atleast_2d(g.B).reshape(A.shape[0], -1)
    C = np.atleast_2d(g.C).reshape(-1, A.shape[0])
    D = np.atleast_2d(g.D)
    Ts = float(getattr(g, "Ts", 0.0))
    if Ts > 0:
        return control.ss(A, B, C, D, Ts)
    return control.ss(A, B, C, D)


def _synthetic_plant():
    """A 4th-order two-resonance stand-in used when the benchmark model has not
    been converted yet."""
    import control

    w1, w2 = 2 * np.pi * 90.0, 2 * np.pi * 240.0
    g1 = control.tf([w1 ** 2], [1, 2 * 0.02 * w1, w1 ** 2])
    g2 = control.tf([w2 ** 2], [1, 2 * 0.015 * w2, w2 ** 2])
    return control.tf2ss(control.minreal(g1 * g2))


def benchmark_plant(channel="Pv", which=(0, 0)):
    """Return ``(P0, label)`` — the benchmark plant if it has been converted,
    otherwise a synthetic stand-in (so the tutorials always run).

    Convert once in MATLAB with ``Examples/private/convert_ident_to_python.m``
    to use the real ``mdl.Pv(1,1)`` model.
    """
    try:
        return load_ident(channel, which), "benchmark 20160829_ident.mat"
    except FileNotFoundError:
        print("[note] ident_python.mat not found -> using a synthetic plant.\n"
              "       Run  Examples/private/convert_ident_to_python.m  in MATLAB\n"
              "       to identify against the real benchmark model.")
        return _synthetic_plant(), "synthetic stand-in"
