import numpy as np
import control
import pytest

import fditools as fdi
from conftest import (true_system, freqresp_on, simulate, make_measurement,
                      mimo_2x2)


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
    for field in ("X", "Y", "sX2", "sY2", "cXY", "sCR", "sG", "FRFn"):
        assert getattr(ud, field) is not None
        assert np.asarray(getattr(ud, field)).shape[0] == Pest.freq.size
    assert ud.nrofp is not None and ud.nrofp >= 2


def test_lpm_periodic_matches_true():
    P0 = true_system()
    fs, df = 2000.0, 1.0
    harm = dict(fs=fs, df=df, fl=5.0, fh=400.0, fr=1.02)
    ms = fdi.multisine(harm, control.tf([1], [1]),
                       dict(itp="r", ctp="n", dtp="f", gtp="l"))
    u = np.tile(np.squeeze(ms.x[0, 0, :]), 6)
    y = simulate(P0, u, fs, noise=1e-5)
    FRF, freq, sG, T = fdi.time2frf_lpm(u, y, fs, order=2, halfwidth=3,
                                        period=ms.nrofs, lines=ms.ex, band=(5, 400))
    true = freqresp_on(P0, freq)
    rel = np.abs(FRF[:, 0] - true) / np.abs(true)
    assert np.median(rel) < 0.05


def test_mimo_ml_orthogonal():
    g, ms, harm, X, Y, freq, true = mimo_2x2(nrofp=6)
    # ML needs the transient removed
    Xl = [fdi.pretreat(X[:, :, e], ms.nrofs, harm["fs"], 1, 0)[0] for e in range(X.shape[2])]
    Yl = [fdi.pretreat(Y[:, :, e], ms.nrofs, harm["fs"], 1, 0)[0] for e in range(Y.shape[2])]
    Pest = fdi.time2frf_ml(np.stack(Xl, 2), np.stack(Yl, 2), ms)
    assert Pest.userdata.method == "orthogonal"
    assert Pest.response.shape == (2, 2, Pest.freq.size)
    assert Pest.userdata.sG.shape == (2, 2, Pest.freq.size)
    rel = np.abs(Pest.response - true) / np.maximum(np.abs(true), 1e-9)
    assert np.median(rel) < 0.02


def test_mimo_lpm_orthogonal_handles_transient():
    # LPM models the transient, so the RAW data (no pretreat) gives a good FRF
    g, ms, harm, X, Y, freq, true = mimo_2x2(nrofp=6)
    Pest = fdi.time2frf_lpm(X, Y, harm["fs"], order=2, halfwidth=3,
                            period=ms.nrofs, lines=ms.ex, band=(5, 400))
    assert Pest.userdata.method == "lpm"
    assert Pest.response.shape == (2, 2, Pest.freq.size)
    rel = np.abs(Pest.response - true) / np.maximum(np.abs(true), 1e-9)
    assert np.median(rel) < 0.02


def test_mimo_lpm_zippered():
    # single zippered experiment: each input owns interleaved lines.
    # smooth (thermal-like) plant so the per-channel 1/nu resolution suffices.
    def lp(k, fc):
        return control.tf([k], [1 / (2 * np.pi * fc), 1])
    g = [[lp(1.0, 20), lp(0.3, 15)], [lp(0.2, 30), lp(1.0, 40)]]
    nu = ny = 2
    fs, df = 2000.0, 1.0
    nrofs = int(fs / df)
    ex = np.arange(1, 201)                          # 0-based excited bins 1..200
    owned = [ex[0::2], ex[1::2]]
    nrofp = 4
    N = nrofp * nrofs
    t = np.arange(N) / fs
    rng = np.random.default_rng(0)
    u = np.zeros((N, nu))
    for i in range(nu):
        for k in owned[i]:
            u[:, i] += np.cos(2 * np.pi * k * df * t + rng.uniform(-np.pi, np.pi))
    y = np.zeros((N, ny))
    for o in range(ny):
        for i in range(nu):
            y[:, o] += control.forced_response(g[o][i], t, u[:, i]).outputs
    Pz = fdi.time2frf_lpm(u, y, fs, order=2, halfwidth=4, period=nrofs,
                          lines=ex, band=(2, 198))
    assert Pz.userdata.method == "lpm"
    assert Pz.response.shape == (ny, nu, Pz.freq.size)
    true = np.array([[[complex(g[o][i](1j * 2 * np.pi * f)) for f in Pz.freq]
                      for i in range(nu)] for o in range(ny)])
    rel = np.abs(Pz.response - true) / np.maximum(np.abs(true), 1e-9)
    assert np.median(rel) < 0.05


def test_lpm_broadband_runs():
    P0 = true_system()
    fs = 2000.0
    rng = np.random.default_rng(4)
    u = rng.standard_normal(4000)
    y = simulate(P0, u, fs, noise=1e-5)
    FRF, freq, sG, T = fdi.time2frf_lpm(u, y, fs, order=2, halfwidth=6,
                                        band=(20, 300))
    true = freqresp_on(P0, freq)
    rel = np.abs(FRF[:, 0] - true) / np.abs(true)
    assert np.median(rel) < 0.2          # broadband, no averaging


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
