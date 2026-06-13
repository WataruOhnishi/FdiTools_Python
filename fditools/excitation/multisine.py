"""Multisine excitation-signal generation (port of ``multisine.m``).

``harm`` and ``options`` may be plain ``dict``s or any object exposing the same
attribute names; the returned :class:`Multisine` mirrors the fields of the
MATLAB output struct (``x``/``time``/``X``/``freq``/``ex``/``cf``/``harm``/
``options``/``Hampl``/``nrofs``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.linalg import hadamard

import control

from ..frfdata import FrfData
from ..aux.misc import f2t
from ..aux.conversions import tfdata
from ._helpers import lin2qlog, randph, schroed, msinl2p, effval, lpnorm


def _ns(obj):
    """Coerce a dict (or namespace) into a mutable ``SimpleNamespace``."""
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    return SimpleNamespace(**{k: getattr(obj, k) for k in vars(obj)})


class Multisine(SimpleNamespace):
    """Designed multisine (attribute container)."""

    def __repr__(self):
        return (f"Multisine(nrofi={self.x.shape[0]}, nrofs={self.nrofs}, "
                f"fl={self.harm.fl:g}, fh={self.harm.fh:g} Hz)")


def _amplitude_count(Hampl):
    """Return ``(list_of_channels, nrofi, is_frd)`` for the amplitude spec."""
    if isinstance(Hampl, FrfData):
        return [Hampl], 1, True
    if isinstance(Hampl, (list, tuple, np.ndarray)):
        return list(Hampl), len(Hampl), False
    return [Hampl], 1, False


def multisine(harm, Hampl, options):
    """Generate a (MIMO) multisine excitation signal.

    Parameters
    ----------
    harm : dict | namespace
        ``fs`` sampling freq [Hz], ``fl``/``fh`` low/high freq [Hz],
        ``df`` resolution [Hz], ``fr`` quasi-log ratio.
    Hampl : control LTI | list of control LTI | FrfData
        Desired input amplitude spectrum per input (one entry per input), or a
        non-parametric :class:`FrfData` weighting.
    options : dict | namespace
        ``itp`` ('s'/'r'), ``ctp`` ('c'/'n'), ``gtp`` ('l'/'q'),
        ``dtp`` ('f'/'o'/'O').

    Returns
    -------
    Multisine
    """
    harm = _ns(harm)
    options = _ns(options)

    channels, nrofi, is_frd = _amplitude_count(Hampl)

    nl = int(np.ceil(harm.fl / harm.df)) + 1          # lowest line number
    nh = int(round(harm.fh / harm.df)) + 1            # highest line number
    nrofs = int(np.ceil(harm.fs / harm.df))           # samples per period

    if nrofi > 1:
        options.itp = "r"
    if np.log2(nrofi) % 1 != 0:
        options.otp = "o"
    else:
        options.otp = "e"

    # ----- excited harmonics (1-based line numbers) -----------------------
    if options.dtp in ("f", "full"):
        ex = np.arange(nl, nh + 1, 1)
    elif options.dtp in ("o", "odd"):
        ex = np.arange(nl, nh + 1, 2)
    elif options.dtp in ("O", "odd-odd"):
        ex = np.arange(nl, nh + 1, 4)
    else:
        raise ValueError(f"unknown dtp={options.dtp!r}")

    idx = np.where(ex <= nl)[0]
    start = int(np.max(idx)) if idx.size else 0
    if options.gtp in ("l", "lin"):
        ex = ex[start:]
    elif options.gtp in ("q", "qlog"):
        ex, _ = lin2qlog(ex[start:], harm.fr)
    else:
        raise ValueError(f"unknown gtp={options.gtp!r}")

    ex0 = ex - 1  # 0-based indices into per-line arrays

    # ----- optimised spectrum R(i, j, line) -------------------------------
    R = np.zeros((nrofi, nrofi, nh), dtype=complex)
    grid = harm.df * np.arange(nrofs // 2)            # df grid in Hz
    w = 1j * 2.0 * np.pi * grid[:nh]
    for i in range(nrofi):
        Xspec = np.zeros(nh, dtype=complex)
        if is_frd:
            frd = channels[i]
            mag_full = np.zeros(nh)
            fr_hz = np.asarray(frd.freq, dtype=float)
            resp = np.abs(frd.response[0, 0, :])
            # place |frd| on the df grid where frequencies coincide
            for kf, fhz in enumerate(grid[:nh]):
                m = np.isclose(fr_hz, fhz, rtol=0, atol=harm.df / 2.0)
                if np.any(m):
                    mag_full[kf] = resp[np.argmax(m)]
            Mag = mag_full
        else:
            Bn, An = tfdata(control.tf(channels[i]))
            Mag = np.abs(np.polyval(Bn, w) / np.polyval(An, w))
        Xspec[ex0] = Mag[ex0]

        for j in range(nrofi):
            if options.itp in ("r", "random"):
                Xj = randph(Xspec, i)          # seed i (== MATLAB i-1, 0-based)
            elif options.itp in ("s", "schroed"):
                Xj = schroed(Xspec)
            else:
                raise ValueError(f"unknown itp={options.itp!r}")

            if options.ctp in ("n", "non-comp"):
                R[i, j, :] = Xj
            elif options.ctp in ("c", "compressed"):
                R[i, j, :] = msinl2p(Xj, nrofs, options.itp)
            else:
                raise ValueError(f"unknown ctp={options.ctp!r}")

    # ----- orthogonal transform ------------------------------------------
    if options.otp in ("e", "even"):
        T = hadamard(nrofi).astype(complex)
    else:
        from ._helpers import orthogonal
        T = orthogonal(nrofi)
    T = np.repeat(T[:, :, None], nh, axis=2)

    # ----- multisine signal ----------------------------------------------
    S = np.zeros((nrofi, nrofi, nrofs), dtype=complex)
    S[:, :, :nh] = R * T
    s = 2.0 * np.real(np.fft.ifft(S, axis=2))
    rms = np.mean(np.abs(s ** 2), axis=2) ** 0.5
    x = s / rms[:, :, None]
    X = np.fft.fft(x, axis=2) / np.sqrt(nrofs)
    X = X[:, :, :nrofs // 2]
    freq = harm.fs * np.arange(nrofs // 2) / nrofs
    time = np.arange(0, 1.0 / harm.df - 0.5 / harm.fs, 1.0 / harm.fs)
    if time.size != nrofs:
        time = np.arange(nrofs) / harm.fs

    # ----- crest factors --------------------------------------------------
    cf = np.zeros((nrofi, nrofi))
    for i in range(nrofi):
        for j in range(nrofi):
            C = S[i, j, :]
            c = f2t(C, nrofs)
            cf[i, j] = lpnorm(c.ravel(), np.inf) / effval(C, ex0)

    return Multisine(
        x=x, time=time, X=X, freq=freq,
        ex=ex0, cf=cf,
        harm=harm, options=options, Hampl=Hampl,
        nrofs=x[0, 0, :].size,
    )
