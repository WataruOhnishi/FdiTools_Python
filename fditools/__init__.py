"""FdiTools — Frequency-Domain System Identification Toolbox (Python port).

Python port of the MATLAB FdiTools toolbox.

Reference
---------
R. Pintelon and J. Schoukens, *System Identification: A Frequency Domain
Approach*, 2nd ed. Wiley-IEEE Press, 2012.

Module map (MATLAB folder -> Python subpackage)
-----------------------------------------------
* ``1_ExcitationDesign``    -> :mod:`fditools.excitation`
* ``2_NonparametricFRF``    -> :mod:`fditools.nonparametric`
* ``3_NonlinearDistortions``-> :mod:`fditools.nonlinear`
* ``4_ParametricEstimation``-> :mod:`fditools.parametric`
* ``5_SelectionValidation`` -> :mod:`fditools.validation`
* ``A_CalculationAuxiliary``-> :mod:`fditools.aux`
"""

from .frfdata import FrfData, UserData

# Excitation design
from .excitation import (multisine, Multisine, sweptsine, prbs, multisine2hdr,
                         effval, lpnorm, lin2qlog, orthogonal, randph, schroed,
                         msinl2p)

# Non-parametric FRF
from .nonparametric import (pretreat, time2frf_ml, time2frf_h1, time2frf_log,
                            splinefit)

# Non-linear distortions
from .nonlinear import time2bla, time2nld

# Parametric estimation
from .parametric import (lsfdi, wlsfdi, nlsfdi, mlfdi, gtlsfdi, btlsfdi, ssfdi,
                         mlfdi_res, nlsfdi_res, btlsfdi_res, fdicost, qsvd)

# Selection / validation
from .validation import chi2test, costtest, residtest

# Calculation auxiliary
from .aux import (ba2theta, theta2ba, ba2hm, hm2ba, tfdata, vectorize_orders,
                  f2t, t2f, dbm, phs, hfrf, cr_rao, fdicohere, fdel_fdi, fcat_fdi)

__version__ = "0.1.0"

__all__ = [
    "FrfData", "UserData",
    "multisine", "Multisine", "sweptsine", "prbs", "multisine2hdr",
    "effval", "lpnorm", "lin2qlog", "orthogonal", "randph", "schroed", "msinl2p",
    "pretreat", "time2frf_ml", "time2frf_h1", "time2frf_log", "splinefit",
    "time2bla", "time2nld",
    "lsfdi", "wlsfdi", "nlsfdi", "mlfdi", "gtlsfdi", "btlsfdi", "ssfdi",
    "mlfdi_res", "nlsfdi_res", "btlsfdi_res", "fdicost", "qsvd",
    "chi2test", "costtest", "residtest",
    "ba2theta", "theta2ba", "ba2hm", "hm2ba", "tfdata", "vectorize_orders",
    "f2t", "t2f", "dbm", "phs", "hfrf", "cr_rao", "fdicohere", "fdel_fdi",
    "fcat_fdi",
]


def bode_fdi(*args, **kwargs):
    """Lazy wrapper around :func:`fditools.aux.plotting.bode_fdi` (needs matplotlib)."""
    from .aux.plotting import bode_fdi as _bode_fdi
    return _bode_fdi(*args, **kwargs)
