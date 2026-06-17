"""Bode-style plotting helper (port of v3.0 ``bode_fdi.m``).

Plots magnitude/phase of one or several FRF/LTI models and, optionally, an
uncertainty curve (e.g. the FRF standard deviation ``sG`` or a noise level) as
an overlaid line or a shaded band.  Kept in a separate module so importing the
toolbox does not require ``matplotlib``.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData


def _local_response(item, fgrid):
    """Return ``(fHz, resp)`` for the (1,1) channel of *item*."""
    if isinstance(item, FrfData):
        return item.freq, item.response[0, 0, :]
    if isinstance(item, tuple):                 # (freq, resp)
        f, r = item
        return np.asarray(f, dtype=float).ravel(), np.asarray(r).ravel()
    import control                              # control LTI

    sys = control.tf(item) if not hasattr(item, "frequency_response") else item
    s = 1j * 2.0 * np.pi * fgrid
    return fgrid, np.array([complex(sys(sk)) for sk in s])


def _wrap_phase(ph, pmin, pmax):
    ph = np.asarray(ph, dtype=float).copy()
    ph = np.where(ph > pmax, ph - 360.0, ph)
    ph = np.where(ph < pmin, ph + 360.0, ph)
    return ph


def _resolve_unc(unc, sys, col):
    """Return ``(uf, um, uname, refsys)`` from the uncertainty specifier."""
    if unc is None:
        return None, None, "uncertainty", None
    if isinstance(unc, str):
        for d in sys:
            if isinstance(d, FrfData) and d.userdata.has(unc):
                v = np.asarray(getattr(d.userdata, unc))
                if v.ndim >= 2:
                    v = v[:, min(col, v.shape[1] - 1)]
                return d.freq, v.ravel(), unc, d
        return None, None, "uncertainty", None
    refsys = next((d for d in sys if isinstance(d, FrfData)), None)
    if isinstance(unc, tuple):                  # (freq, mag)
        f, m = unc
        return np.asarray(f).ravel(), np.asarray(m).ravel(), "uncertainty", refsys
    arr = np.asarray(unc)
    if arr.ndim == 2 and arr.shape[1] == 2:     # [freq mag]
        return arr[:, 0], arr[:, 1], "uncertainty", refsys
    uf = refsys.freq if refsys is not None else None
    return uf, arr.ravel(), "uncertainty", refsys


def bode_fdi(sys, unc=None, sigma=1.0, style="line", col=0, legend=None,
             unit="Hz", xlim=None, maglim=None, title=None,
             pmin=-180.0, pmax=180.0, legendloc="best"):
    """Bode plot of FRF(s) with optional uncertainty (line or band).

    Parameters
    ----------
    sys : FrfData | control LTI | (freq, resp) | list of those
        Curves to draw (only the (1,1) channel of a MIMO model is shown).
    unc : None | str | (freq, mag) | array
        Uncertainty source: a ``UserData`` field name (``'sG'``, ``'sCR'``,
        ``'FRFn'`` ...) searched over the systems, an explicit ``(freq, mag)``
        pair / ``Nx2`` array, or a magnitude vector on the first FRF's grid.
    sigma : float        multiplier applied to the uncertainty (default 1)
    style : 'line' | 'band'
        ``line`` overlays ``20*log10(sigma*unc)``; ``band`` shades
        ``|G_ref| +/- sigma*unc`` around the owning system.
    col : int            column of a multi-output UserData field (default 0)
    legend : list[str]   names; an ``(N+1)``-th entry names the uncertainty
    unit, xlim, maglim, title, pmin, pmax, legendloc : plot cosmetics

    Returns
    -------
    fig, (axM, axP) : matplotlib objects
    """
    import matplotlib.pyplot as plt

    if not isinstance(sys, list):
        sys = [sys]
    N = len(sys)

    # frequency grid for tf/ss systems: span the FRF frequencies if present
    ffrd = []
    for d in sys:
        if isinstance(d, FrfData):
            ffrd.append(np.asarray(d.freq, dtype=float))
        elif isinstance(d, tuple):
            ffrd.append(np.asarray(d[0], dtype=float))
    if xlim is not None:
        fdef = list(xlim)
    elif ffrd:
        allf = np.concatenate(ffrd)
        allf = allf[allf > 0]
        fdef = [allf.min(), allf.max()]
    else:
        fdef = [1.0, 1e4]
    fgrid = np.logspace(np.log10(fdef[0]), np.log10(fdef[1]), 600)

    fig, (axM, axP) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.2))

    mag_all = []
    for k, d in enumerate(sys):
        fHz, resp = _local_response(d, fgrid)
        name = legend[k] if (legend is not None and k < len(legend)) else f"G{k + 1}"
        magdb = 20.0 * np.log10(np.abs(resp))
        mag_all.append(magdb[np.isfinite(magdb)])
        axM.semilogx(fHz, magdb, label=name)
        axP.semilogx(fHz, _wrap_phase(np.angle(resp) * 180.0 / np.pi, pmin, pmax),
                     label=name)

    uf, um, uname, refsys = _resolve_unc(unc, sys, col)
    unc_named = legend is not None and len(legend) >= N + 1
    if unc_named:
        uname = legend[N]
    if um is not None:
        um = sigma * np.abs(np.asarray(um, dtype=float).ravel())
        uf = np.asarray(uf, dtype=float).ravel()
        if style == "line":
            axM.semilogx(uf, 20.0 * np.log10(um), "--",
                         label=uname if unc_named else None)
        elif style == "band":
            if refsys is None:
                axM.semilogx(uf, 20.0 * np.log10(um), "--",
                             label=uname if unc_named else None)
            else:
                rf, rresp = _local_response(refsys, uf)
                gmag = np.interp(uf, rf, np.abs(rresp))
                up = 20.0 * np.log10(gmag + um)
                lo = 20.0 * np.log10(np.maximum(gmag - um, np.finfo(float).eps))
                axM.fill_between(uf, lo, up, color=(0.6, 0.6, 0.6), alpha=0.25,
                                 label=uname if unc_named else None)
        else:
            raise ValueError("style must be 'line' or 'band'.")

    axM.set_ylabel("Magnitude [dB]")
    axP.set_ylabel("Phase [deg]")
    axP.set_xlabel(f"Frequency [{unit}]")
    axP.set_yticks(np.arange(pmin, pmax + 1, 90))
    axP.set_ylim(pmin, pmax)
    axM.set_xlim(fdef)
    axP.set_xlim(fdef)
    if maglim is not None:
        axM.set_ylim(maglim)
    elif mag_all:
        m = np.concatenate(mag_all)
        if m.size:
            axM.set_ylim(np.floor((m.min() - 10) / 10) * 10,
                         np.ceil((m.max() + 10) / 10) * 10)
    if title:
        axM.set_title(title)
    if legend is not None:
        axM.legend(loc=legendloc, fontsize=8)
    fig.tight_layout()
    return fig, (axM, axP)
