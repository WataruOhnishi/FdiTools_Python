"""Export a (SISO) multisine reference vector to a C header file
(port of ``multisine2hdr.m``)."""

from __future__ import annotations

import os

import numpy as np

from ..frfdata import FrfData
from ..auxiliary.conversions import tfdata


def _peak2rms(sig):
    return np.max(np.abs(sig)) / np.sqrt(np.mean(sig ** 2))


def multisine2hdr(ms, fname, fmt="d"):
    """Write the multisine time signal as a C array.

    Parameters
    ----------
    ms : Multisine
    fname : str
        Output path, e.g. ``'data/ms.h'``.
    fmt : {'d','f','m'}
        ``d`` double / ``f`` float / ``m`` MyWay PE-Expert3 ``far float``.

    Returns
    -------
    path : str
    """
    folder = os.path.dirname(fname)
    name = os.path.splitext(os.path.basename(fname))[0]
    ext = os.path.splitext(fname)[1]
    if folder:
        os.makedirs(folder, exist_ok=True)

    signal = np.squeeze(np.asarray(ms.x))
    nrofs = signal.shape[0]
    harm = ms.harm

    lines = []
    lines.append(f"// CREST FACTOR = {_peak2rms(signal):f}")
    lines.append(f"// SIGNAL LENGTH = {nrofs * 1e3 / harm.fs:g} [ms]\n")
    lines.append("/* HARMONICS PARAMETERS ")
    lines.append("** ----------------------- ")
    lines.append(f"** fs = {harm.fs:f} [Hz]: sampling frequency ")
    lines.append(f"** df = {harm.df:f} [Hz]: frequency resolution ")
    lines.append(f"** fl = {harm.fl:f} [Hz]: lowest frequency ")
    lines.append(f"** fh = {harm.fh:f} [Hz]: highest frequency ")
    if getattr(ms.options, "gtp", None) == "q":
        lines.append(f"** fr = {harm.fr:f} : frequency log ratio ")
    lines.append("*/ \n")
    lines.append("/* DESIGN OPTIONS ")
    lines.append("** ----------------------- ")
    lines.append(f"** itp = {ms.options.itp} : init phase type ")
    lines.append(f"** ctp = {ms.options.ctp} : compression type ")
    lines.append(f"** dtp = {ms.options.dtp} : signal type ")
    lines.append(f"** gtp = {ms.options.gtp} : grid type ")
    lines.append("*/ \n")

    if isinstance(ms.Hampl, FrfData):
        lines.append("/* NON-PARAMETRIC WEIGHTING */ \n")
    else:
        chan = ms.Hampl[0] if isinstance(ms.Hampl, (list, tuple)) else ms.Hampl
        num, den = tfdata(chan)
        lines.append("/* AMPLITUDE SPECTRUM ")
        lines.append("** ----------------------- ")
        lines.append("** num = [ " + " ".join(f"{v:e}" for v in num) + " ]")
        lines.append("** den = [ " + " ".join(f"{v:e}" for v in den) + " ]")
        lines.append("*/ \n")

    if fmt == "d":
        ctype, prec = "double", 16
    elif fmt == "f":
        ctype, prec = "float", 8
    elif fmt == "m":
        ctype, prec = "far float", 8
    else:
        raise ValueError("format must be 'd', 'f' or 'm'")

    array_name = f"refvec_{name}"
    nrofs_name = f"NROFS_{name}"
    body = [f"#define {nrofs_name} {nrofs} ",
            f"{ctype} {array_name} [{nrofs_name}] = {{ "]
    row = []
    for k, v in enumerate(signal, 1):
        row.append(f"{v:.{prec}e}, ")
        if k % 10 == 0:
            body.append("".join(row))
            row = []
    if row:
        body.append("".join(row))
    body.append("}; ")

    path = os.path.join(folder, name + ext) if folder else name + ext
    with open(path, "w") as fid:
        fid.write("\n".join(lines) + "\n" + "\n".join(body) + "\n")
    return path
