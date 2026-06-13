"""Bode-style plotting helper (port of ``bode_fdi.m``).

Kept in a separate module so importing the toolbox does not require
``matplotlib``.  ``bode_fdi`` accepts a list mixing :class:`FrfData`,
``control`` LTI systems and ``(freq_Hz, complex_response)`` tuples.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData


def _as_freq_resp(item, freq_default):
    """Return ``(freq_Hz, complex_response, userdata_or_None)``."""
    if isinstance(item, FrfData):
        resp = item.response[0, 0, :]
        return item.freq, resp, item.userdata
    if isinstance(item, tuple):
        f, r = item
        return np.asarray(f, dtype=float).ravel(), np.asarray(r).ravel(), None
    # assume a control LTI object
    import control

    sys = control.tf(item) if not hasattr(item, "frequency_response") else item
    s = 1j * 2.0 * np.pi * freq_default
    resp = np.array([complex(sys(sk)) for sk in s])
    return freq_default, resp, None


def bode_fdi(data, noise=None, pmin=-180.0, pmax=180.0, title=None, labels=None):
    """Plot magnitude/phase (and coherence when available).

    Parameters
    ----------
    data : FrfData | control system | list of those
        Curves to draw.  A bare ``(freq, resp)`` tuple is also accepted.
    noise : str | (freq, resp) | None
        Either the name of a UserData noise field (e.g. ``'FRFn'``,
        ``'sGhat'``) to overlay, or an explicit ``(freq, resp)`` pair.
    pmin, pmax : float
        Phase axis limits (deg); phase is wrapped into this band.
    title : str, optional
    labels : list[str], optional
        Legend labels for the magnitude curves.

    Returns
    -------
    fig, axes : matplotlib objects
    """
    import matplotlib.pyplot as plt

    if not isinstance(data, (list, tuple)) or (
        isinstance(data, tuple) and len(data) == 2 and np.isscalar(data[0]) is False
        and np.ndim(data[0]) >= 1 and not isinstance(data[0], (list, tuple))
        and np.iscomplexobj(np.asarray(data[1]))
    ):
        # single (freq, resp) tuple or single object
        if isinstance(data, tuple):
            data = [data]
        else:
            data = [data]
    data = list(data)

    freq_default = np.logspace(0, 3, 400)

    # detect coherence availability
    coh_flag = any(isinstance(d, FrfData) and d.userdata.has("cxy") for d in data)
    ms_flag = any(isinstance(d, FrfData) and d.userdata.has("sGhat") for d in data)
    if ms_flag:
        coh_flag = False
    nplot = 3 if coh_flag else 2

    fig, axes = plt.subplots(nplot, 1, sharex=True, figsize=(7, 2.6 * nplot))
    ax_mag, ax_ph = axes[0], axes[1]

    curves = [_as_freq_resp(d, freq_default) for d in data]

    for k, (f, r, _ud) in enumerate(curves):
        lbl = labels[k] if labels and k < len(labels) else None
        ax_mag.semilogx(f, 20.0 * np.log10(np.abs(r)), label=lbl)
    ax_mag.set_ylabel("Magnitude [dB]")
    if title:
        ax_mag.set_title(title)
    if labels:
        ax_mag.legend(loc="best", fontsize=8)

    for f, r, _ud in curves:
        ph = np.angle(r) * 180.0 / np.pi
        ph = np.where(ph > pmax, ph - 360.0, ph)
        ph = np.where(ph < pmin, ph + 360.0, ph)
        ax_ph.semilogx(f, ph)
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_ylim(pmin, pmax)
    ax_ph.set_yticks(np.arange(pmin, pmax + 1, 90))

    if coh_flag:
        ax_coh = axes[2]
        for d in data:
            if isinstance(d, FrfData) and d.userdata.has("cxy"):
                ax_coh.semilogx(d.freq, np.asarray(d.userdata.cxy)[:, 0])
        ax_coh.set_ylabel("Coherence [-]")
        ax_coh.set_xlabel("Frequency [Hz]")
    else:
        ax_ph.set_xlabel("Frequency [Hz]")
        # overlay noise on the magnitude axis
        if isinstance(noise, str):
            for d in data:
                if isinstance(d, FrfData) and d.userdata.has(noise):
                    val = np.asarray(getattr(d.userdata, noise))
                    ax_mag.semilogx(d.freq, 20.0 * np.log10(np.abs(val[:, 0])),
                                    "--", linewidth=0.8)
        elif noise is not None:
            f, r = noise
            ax_mag.semilogx(np.asarray(f).ravel(),
                            20.0 * np.log10(np.abs(np.asarray(r).ravel())),
                            "--", linewidth=0.8)

    fig.tight_layout()
    return fig, axes
