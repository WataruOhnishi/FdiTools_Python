"""Calculation-auxiliary helpers (port of ``src/A_CalculationAuxiliary``)."""

from .conversions import ba2theta, theta2ba, ba2hm, hm2ba, tfdata, vectorize_orders
from .misc import f2t, t2f, dbm, phs
from .frfutils import hfrf, cr_rao, fdicohere, fdel_fdi, fcat_fdi, frfconf

__all__ = [
    "ba2theta", "theta2ba", "ba2hm", "hm2ba", "tfdata", "vectorize_orders",
    "f2t", "t2f", "dbm", "phs",
    "hfrf", "cr_rao", "fdicohere", "fdel_fdi", "fcat_fdi", "frfconf",
]
