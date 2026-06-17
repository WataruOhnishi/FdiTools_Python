import numpy as np
import control
import pytest

import fditools as fdi
from conftest import true_system, simulate, freqresp_on, mimo_2x2


def test_time2bla_matches_frf_for_linear_system():
    P0 = true_system()
    fs, df = 2000.0, 1.0
    harm = dict(fs=fs, df=df, fl=5.0, fh=300.0, fr=1.02)
    opt = dict(itp="r", ctp="n", dtp="f", gtp="l")
    # several independent realisations (different random phases via seed)
    cols_u, cols_y = [], []
    for r in range(3):
        ms = fdi.multisine(harm, control.tf([1], [1]), opt)
        one = np.squeeze(ms.x[0, 0, :])
        u = np.tile(one, 4)
        y = simulate(P0, u, fs, seed=r, noise=1e-5)
        xp, _ = fdi.pretreat(u, ms.nrofs, fs, 1, 0)
        yp, _ = fdi.pretreat(y, ms.nrofs, fs, 1, 0)
        cols_u.append(xp)
        cols_y.append(yp)
    X, Y, FRF, freq, Gbla, sX2, sY2, cXY = fdi.time2bla(
        np.column_stack(cols_u), np.column_stack(cols_y), fs, 5, 300, df)
    true = freqresp_on(P0, freq)
    rel = np.abs(FRF - true) / np.abs(true)
    assert np.median(rel) < 0.1
    # nonlinear std (row 4) should be small relative to the response for a
    # linear system
    assert np.median(np.abs(Gbla[3])) < np.median(np.abs(Gbla[0]))


def test_time2bla_mimo_linear():
    # M realizations (each = nu orthogonal experiments), independent noise.
    M, nu = 3, 2
    Xl, Yl = [], []
    true = None
    for m in range(M):
        g, ms, harm, X, Y, freq, tr = mimo_2x2(nrofp=6, seed=m, noise=1e-4)
        true = tr
        for e in range(X.shape[2]):
            Xl.append(fdi.pretreat(X[:, :, e], ms.nrofs, harm["fs"], 1, 0)[0])
            Yl.append(fdi.pretreat(Y[:, :, e], ms.nrofs, harm["fs"], 1, 0)[0])
    bla = fdi.time2bla_mimo(np.stack(Xl, 2), np.stack(Yl, 2), ms, M)
    assert bla["G"].response.shape == (2, 2, bla["freq"].size)
    rel = np.abs(bla["G"].response - true) / np.maximum(np.abs(true), 1e-9)
    assert np.median(rel) < 0.05
    # linear plant: nonlinear-distortion level stays near the noise level
    assert np.median(bla["sG_nl"]) < 5 * np.median(bla["sG_noise"])


def test_time2nld_runs():
    P0 = true_system()
    fs, df = 2000.0, 1.0
    harm = dict(fs=fs, df=df, fl=5.0, fh=200.0, fr=1.02)
    opt = dict(itp="r", ctp="n", dtp="O", gtp="l")   # odd-odd grid
    ms = fdi.multisine(harm, control.tf([1], [1]), opt)
    one = np.squeeze(ms.x[0, 0, :])
    u = np.tile(one, 6)
    y = simulate(P0, u, fs, noise=1e-5)
    out = fdi.time2nld(u, y, fs, 5, 200, df)
    Yl, freql, Yo, freqo, Ye, freqe, Yn, freqn = out
    assert freql.size > 0
    assert Yn.shape[0] == freqn.size
