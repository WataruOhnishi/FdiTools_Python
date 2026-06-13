"""Excitation design (port of ``src/1_ExcitationDesign``)."""

from .multisine import multisine, Multisine
from .sweptsine import sweptsine
from .prbs import prbs
from .multisine2hdr import multisine2hdr
from ._helpers import (effval, lpnorm, lin2qlog, orthogonal, randph, schroed,
                       msinl2p)

__all__ = [
    "multisine", "Multisine", "sweptsine", "prbs", "multisine2hdr",
    "effval", "lpnorm", "lin2qlog", "orthogonal", "randph", "schroed", "msinl2p",
]
