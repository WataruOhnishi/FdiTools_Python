import numpy as np
import control
import pytest

import fditools as fdi
from conftest import true_system, freqresp_on, simulate, make_measurement


def test_pretreat_removes_transient_and_offset():
    nrofs = 100
    period = np.sin(2 * np.pi * np.arange(nrofs) / nrofs)
    sig = np.tile(period, 4) + 3.0          # constant offset
    y, t = fdi.pretreat(sig, nrofs, fs=100.0, nroft=1, trend=0)
    assert y.size == nrofs * 3
    assert abs(np.mean(y)) < 1e-9           # offset removed


def test_frf_matches_true_linear(measurement):
    P0, ms, Pest = measurement
    true = freqresp_on(P0, Pest.freq)
    est = Pest.response[0, 0, :]
    rel = np.abs(est - true) / np.abs(true)
    assert np.median(rel) < 0.05


def test_frf_matches_true_qlog():
    P0, ms, Pest = make_measurement(gtp="q", nrofp=8)
    true = freqresp_on(P0, Pest.freq)
    est = Pest.response[0, 0, :]
    rel = np.abs(est - true) / np.abs(true)
    assert np.median(rel) < 0.06
    assert Pest.freq.size == ms.ex.size


def test_userdata_populated(measurement):
    _, _, Pest = measurement
    ud = Pest.userdata
    for field in ("X", "Y", "sX2", "sY2", "cXY", "sCR", "sGhat", "FRFn"):
        assert getattr(ud, field) is not None
        assert np.asarray(getattr(ud, field)).shape[0] == Pest.freq.size


def test_time2frf_h1_rough_match():
    P0 = true_system()
    fs, df = 2000.0, 1.0
    nrofs = int(fs / df)
    rng = np.random.default_rng(2)
    u = rng.standard_normal(nrofs * 8)
    y = simulate(P0, u, fs, noise=1e-5)
    X2, Y2, FRF, freq, coh = fdi.time2frf_h1(u, y, fs, 20, 300, df, window=1)
    true = freqresp_on(P0, freq)
    rel = np.abs(FRF[:, 0] - true) / np.abs(true)
    # H1 on random noise is noisier; require a loose median bound
    assert np.median(rel) < 0.25
