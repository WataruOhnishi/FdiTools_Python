import numpy as np
import control
import pytest

import fditools as fdi
from conftest import freqresp_on


def _model_err(Hm, freq, true):
    sys = control.tf(Hm[0, 0])
    fit = np.array([complex(sys(1j * 2 * np.pi * f)) for f in freq])
    return np.median(np.abs(fit - true) / np.abs(true))


def test_conversions_roundtrip():
    n, mh, ml = 4, 2, 0
    An = np.array([1.0, 2.0, 3.0, 2.5, 1.0])
    Bn = np.array([[0.0, 0.0, 5.0, 4.0, 1.0]])     # numerator order 2
    theta = fdi.ba2theta(Bn, An, n, mh, ml)
    Bn2, An2 = fdi.theta2ba(theta, n, mh, ml)
    assert np.allclose(An2, An)
    assert np.allclose(Bn2, Bn)


def test_ba2hm_hm2ba_roundtrip():
    An = np.array([1.0, 0.5, 2.0])
    Bn = np.array([[0.0, 0.0, 3.0]])
    Hm = fdi.ba2hm(Bn, An, 1, 1)
    Bn2, An2 = fdi.hm2ba(Hm)
    assert np.allclose(An2 / An2[0], An / An[0])
    assert np.allclose(Bn2[0] / An2[0], Bn[0] / An[0])


@pytest.mark.parametrize("which", ["ls", "wls", "ml", "nls", "gtls", "btls"])
def test_estimators_recover_system(measurement, which):
    P0, ms, Pest = measurement
    freq = Pest.freq
    true = freqresp_on(P0, freq)
    n, mh, ml = 2, 0, 0
    FRF_W = np.ones(freq.size)

    if which == "ls":
        Hm, _ = fdi.lsfdi(Pest.userdata.X, Pest.userdata.Y, freq, n, mh, ml,
                          "c", ms.harm.fs)
    elif which == "wls":
        _, Hm = fdi.nlsfdi(Pest, FRF_W, n, mh, ml, 200, 1e-10, 0, "c")
    elif which == "ml":
        Hm, _ = fdi.mlfdi(Pest, n, mh, ml, 200, 1e-10, 0, "c")
    elif which == "nls":
        Hm, _ = fdi.nlsfdi(Pest, FRF_W, n, mh, ml, 200, 1e-10, 0, "c")
    elif which == "gtls":
        _, Hm = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 200, 1e-10, "c")
    elif which == "btls":
        Hm, _ = fdi.btlsfdi(Pest, n, mh, ml, 1.0, 200, 1e-10, "c")

    assert _model_err(Hm, freq, true) < 0.05


def test_classical_and_structured_mlfdi_agree(measurement):
    P0, ms, Pest = measurement
    freq = Pest.freq
    ud = Pest.userdata
    n, mh, ml = 2, 0, 0
    Hs, _ = fdi.mlfdi(Pest, n, mh, ml, 100, 1e-10, 0, "c")
    Hc, _ = fdi.mlfdi(ud.X, ud.Y, freq, ud.sX2, ud.sY2, ud.cXY,
                      n, mh, ml, 100, 1e-10, 0, "c", ms.harm.fs)
    bs, as_ = fdi.hm2ba(Hs)
    bc, ac = fdi.hm2ba(Hc)
    assert np.allclose(as_, ac, atol=1e-8)
    assert np.allclose(bs, bc, atol=1e-8)


def test_hfrf_matches_control(measurement):
    _, _, Pest = measurement
    Hm, _ = fdi.mlfdi(Pest, 2, 0, 0, 100, 1e-10, 0, "c")
    freq = Pest.freq
    FRF = fdi.hfrf(Hm, freq)
    sys = control.tf(Hm[0, 0])
    ref = np.array([complex(sys(1j * 2 * np.pi * f)) for f in freq])
    assert np.allclose(FRF[:, 0], ref, rtol=1e-8, atol=1e-8)
