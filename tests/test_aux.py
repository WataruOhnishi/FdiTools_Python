import numpy as np
import control
import pytest

import fditools as fdi
from conftest import true_system, simulate, freqresp_on, make_measurement


def test_frfconf():
    # large-M / unspecified -> sqrt(-log(1-p))
    assert np.isclose(fdi.frfconf(0.95), np.sqrt(-np.log(0.05)))
    # finite M gives a (larger) finite positive factor that ->limit as M grows
    f20 = fdi.frfconf(0.95, 20)
    f1e6 = fdi.frfconf(0.95, 1_000_000)
    assert f20 > fdi.frfconf(0.95)              # F-quantile > chi limit
    assert np.isclose(f1e6, np.sqrt(-np.log(0.05)), atol=1e-3)
    with pytest.raises(ValueError):
        fdi.frfconf(1.0)


def test_dbm_and_phs():
    x = np.array([1.0, 10.0, 100.0])
    assert np.allclose(fdi.dbm(x), [0.0, 20.0, 40.0])
    z = np.exp(1j * np.linspace(0, np.pi, 50))
    ph = fdi.phs(z)
    assert ph.shape == z.shape
    assert np.all(np.isfinite(ph))


def test_cr_rao_shape_and_symmetry(measurement):
    P0, ms, Pest = measurement
    ud = Pest.userdata
    Hm, _ = fdi.mlfdi(Pest, 2, 0, 0, 100, 1e-10, 0, "c")
    Bn, An = fdi.hm2ba(Hm)
    F = fdi.cr_rao(ud.X[:, 0], ud.Y[:, 0], Pest.freq, Bn[0], An,
                   ud.sX2[:, 0], ud.sY2[:, 0], ud.cXY[:, 0],
                   2, 0, 0, "c", ms.harm.fs)
    assert F.shape == (3, 3)                 # n + (mh-ml+1) = 2 + 1
    assert np.allclose(F, F.T)


def test_fdel_and_fcat(measurement):
    P0, ms, Pest = measurement
    n0 = Pest.freq.size
    low = fdi.fdel_fdi(Pest, 200.0, 400.0)   # keep 5..199
    high = fdi.fdel_fdi(Pest, 5.0, 199.0)    # keep 200..400
    assert low.freq.size + high.freq.size == n0
    merged = fdi.fcat_fdi(low, high)
    assert merged.freq.size == n0
    assert np.all(np.diff(merged.freq) > 0)  # sorted ascending


def test_time2frf_log_rough_match():
    P0 = true_system()
    fs, df = 2000.0, 1.0
    nrofs = int(fs / df)
    rng = np.random.default_rng(3)
    u = rng.standard_normal(nrofs * 8)
    y = simulate(P0, u, fs, noise=1e-5)
    X, Y, FRF, freq, coh = fdi.time2frf_log(u, y, fs, 30, 280, df, window=1)
    true = freqresp_on(P0, freq)
    rel = np.abs(FRF[:, 0] - true) / np.abs(true)
    assert np.median(rel) < 0.3


def test_splinefit_fits_line():
    x = np.linspace(0, 10, 200)
    y = 2 * x + 1 + 0.01 * np.random.default_rng(0).standard_normal(x.size)
    spl = fdi.splinefit(x, y, 5, order=4)
    yr = spl(x)
    assert np.max(np.abs(yr - (2 * x + 1))) < 0.1


def test_ssfdi_runs_and_shapes(measurement):
    P0, ms, Pest = measurement
    FRF = Pest.frf_columns()                 # (nroff, 1)
    FRF_W = np.ones_like(FRF, dtype=float)
    A, B, C, D, fs = fdi.ssfdi(FRF, FRF_W, Pest.freq, 1, 1, 8, 8, order=2)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)
    assert D.shape == (1, 1)
    assert np.all(np.isfinite(A))
