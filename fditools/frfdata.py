"""Frequency-response data container (Python analogue of the MATLAB ``frd``
object enriched with ``UserData``).

In the MATLAB toolbox the estimated non-parametric model is a Control System
Toolbox ``frd`` object whose ``UserData`` struct carries the raw spectra, the
noise (co)variances, the Cramer-Rao bound and the multisine description.  The
parametric estimators (``mlfdi``/``nlsfdi``/``btlsfdi`` ...) accept either that
structured object or the classical positional argument list.

:class:`FrfData` reproduces the pieces that the rest of the toolbox relies on:

* ``freq``      : frequency lines in **Hz** (1-D array)
* ``response``  : complex FRF, shape ``(nrofo, nrofi, nroff)``
* ``userdata``  : :class:`UserData` namespace (``X``, ``Y``, ``sX2`` ...)

It is intentionally lightweight; for plotting / ``freqresp`` interoperability
use :meth:`FrfData.to_frd` to obtain a genuine ``control.FrequencyResponseData``.
"""

from __future__ import annotations

import numpy as np


class UserData:
    """Mutable attribute bag mirroring MATLAB's ``frd.UserData`` struct.

    Unknown fields simply do not exist until assigned, matching the dynamic
    behaviour of the MATLAB code which adds fields as they are computed.
    """

    # v3.0: ``sG`` is the FRF standard deviation (PS2012 eq.2-38, = sqrt(2)*sCR);
    # ``nrofp`` is the number of averaged periods M (for confidence bounds).
    __slots__ = (
        "X", "Y", "FRFn", "sX2", "sY2", "cXY", "sCR", "sG", "nrofp",
        "ms", "cxy", "x", "y", "method", "T",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, None)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def has(self, name) -> bool:
        """Return True if field *name* exists and is not ``None``.

        Equivalent to MATLAB ``isfield(UserData, name)`` for the purposes of
        this toolbox (a field that was never populated stays ``None``).
        """
        return getattr(self, name, None) is not None


class FrfData:
    """Container holding a measured/estimated FRF and its metadata.

    Parameters
    ----------
    response : array_like
        Complex FRF.  Accepted shapes: ``(nroff,)`` (SISO),
        ``(nrofo, nrofi, nroff)`` (MATLAB ``frd`` convention) or
        ``(nroff, nrofh)`` (column-stacked, ``h = (i-1)*nrofo + o``); in the
        last case pass *nrofi*/*nrofo* explicitly.
    freq : array_like
        Frequency lines in Hz.
    nrofi, nrofo : int, optional
        Number of inputs / outputs.  Required only when *response* is given in
        the column-stacked ``(nroff, nrofh)`` layout.
    """

    def __init__(self, response, freq, nrofi=None, nrofo=None, userdata=None):
        freq = np.asarray(freq, dtype=float).ravel()
        nroff = freq.size
        resp = np.asarray(response)

        if resp.ndim == 1:
            response3 = resp.reshape(1, 1, nroff)
        elif resp.ndim == 3:
            response3 = resp
        elif resp.ndim == 2:
            # column-stacked (nroff, nrofh) -> (nrofo, nrofi, nroff)
            if resp.shape[0] != nroff:
                resp = resp.T
            nrofh = resp.shape[1]
            if nrofi is None and nrofo is None:
                nrofi, nrofo = 1, nrofh
            elif nrofi is None:
                nrofi = nrofh // nrofo
            elif nrofo is None:
                nrofo = nrofh // nrofi
            response3 = np.empty((nrofo, nrofi, nroff), dtype=complex)
            for h in range(nrofh):
                i = h // nrofo  # 0-based input index
                o = h - i * nrofo  # 0-based output index
                response3[o, i, :] = resp[:, h]
        else:
            raise ValueError("response must be 1-D, 2-D or 3-D")

        self.freq = freq
        self.response = np.asarray(response3, dtype=complex)
        self.userdata = userdata if userdata is not None else UserData()

    # ------------------------------------------------------------------ #
    # shape helpers
    # ------------------------------------------------------------------ #
    @property
    def nrofo(self) -> int:
        return self.response.shape[0]

    @property
    def nrofi(self) -> int:
        return self.response.shape[1]

    @property
    def nroff(self) -> int:
        return self.response.shape[2]

    @property
    def resp(self):
        """Alias of :attr:`response` (matches MATLAB ``Pest.resp``)."""
        return self.response

    @property
    def UserData(self):
        """Alias of :attr:`userdata` (matches MATLAB ``Pest.UserData``)."""
        return self.userdata

    @property
    def frequency(self):
        """Angular-frequency-agnostic alias used by the plotting helper."""
        return self.freq

    # ------------------------------------------------------------------ #
    # conversions
    # ------------------------------------------------------------------ #
    def frf_columns(self):
        """Return the FRF as a ``(nroff, nrofh)`` matrix.

        Column ``h = (i-1)*nrofo + o`` (1-based) holds :math:`H_{oi}`, the same
        ordering used throughout the parametric estimators.
        """
        nrofo, nrofi, nroff = self.response.shape
        out = np.empty((nroff, nrofi * nrofo), dtype=complex)
        for i in range(nrofi):
            for o in range(nrofo):
                out[:, i * nrofo + o] = self.response[o, i, :]
        return out

    def to_frd(self):
        """Return a ``control.FrequencyResponseData`` (omega in rad/s)."""
        import control

        return control.frd(self.response, 2.0 * np.pi * self.freq)

    def __getitem__(self, key):
        """``Pest[o, i]`` -> single-channel :class:`FrfData` (0-based)."""
        if not isinstance(key, tuple):
            key = (key, 0)
        o, i = key
        sub = self.response[o:o + 1, i:i + 1, :]
        return FrfData(sub, self.freq, userdata=self.userdata)

    def __repr__(self):
        return (f"FrfData(nrofo={self.nrofo}, nrofi={self.nrofi}, "
                f"nroff={self.nroff}, f=[{self.freq[0]:g}..{self.freq[-1]:g}] Hz)")
