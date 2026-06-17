"""Shared pytest fixtures and helpers for the FdiTools test suite."""

import os
import sys

import numpy as np
import control
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fditools as fdi  # noqa: E402


def true_system():
    """A lightly damped 2nd-order resonance used across tests."""
    wn = 2 * np.pi * 120.0
    zeta = 0.02
    return control.tf([wn ** 2], [1.0, 2 * zeta * wn, wn ** 2])


def freqresp_on(sys, freq_hz):
    return np.array([complex(sys(1j * 2 * np.pi * f)) for f in freq_hz])


def simulate(sys, u, fs, seed=0, noise=1e-4):
    T = np.arange(u.size) / fs
    y = control.forced_response(sys, T, u).outputs
    rng = np.random.default_rng(seed)
    return y + noise * rng.standard_normal(y.shape)


def make_measurement(gtp="l", ctp="n", fh=400.0, nrofp=6, noise=1e-4):
    """Design a multisine, simulate the true system and estimate the FRF."""
    P0 = true_system()
    harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=fh, fr=1.02)
    options = dict(itp="r", ctp=ctp, dtp="f", gtp=gtp)
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    one = np.squeeze(ms.x[0, 0, :])
    u = np.tile(one, nrofp)
    y = simulate(P0, u, harm["fs"], noise=noise)
    xp, _ = fdi.pretreat(u, ms.nrofs, harm["fs"], 1, 0)
    yp, _ = fdi.pretreat(y, ms.nrofs, harm["fs"], 1, 0)
    Pest = fdi.time2frf_ml(xp, yp, ms)
    return P0, ms, Pest


@pytest.fixture(scope="module")
def measurement():
    return make_measurement()


@pytest.fixture(scope="module")
def P0():
    return true_system()


def mimo_2x2(nrofp=6, seed=None, noise=0.0):
    """A 2x2 modal plant + 2-input multisine, simulated as ``(N, 2, ne)`` data.

    Returns ``(g, ms, harm, X, Y, freq, true)`` where *g[o][i]* are the SISO
    channels, *X/Y* the (N, 2, ne) experiment arrays, and *true* the (2,2,nl)
    FRF on the excited lines.  Output noise (per sample) is added when *noise*>0.
    """
    w1, w2 = 2 * np.pi * 60, 2 * np.pi * 150
    g = [[control.tf([w1 ** 2], [1, 2 * 0.02 * w1, w1 ** 2]),
          control.tf([0.3 * w1 ** 2], [1, 2 * 0.05 * w1, w1 ** 2])],
         [control.tf([0.2 * w2 ** 2], [1, 2 * 0.04 * w2, w2 ** 2]),
          control.tf([w2 ** 2], [1, 2 * 0.03 * w2, w2 ** 2])]]
    harm = dict(fs=5000.0, df=1.0, fl=5.0, fh=400.0, fr=1.02)
    ms = fdi.multisine(harm, [control.tf([1], [1]), control.tf([1], [1])],
                       dict(itp="r", ctp="n", dtp="f", gtp="l"))
    nu = ny = 2
    ne = ms.x.shape[1]
    N = nrofp * ms.nrofs
    T = np.arange(N) / harm["fs"]
    rng = np.random.default_rng(seed) if seed is not None else None
    X = np.zeros((N, nu, ne))
    Y = np.zeros((N, ny, ne))
    for e in range(ne):
        Ue = np.tile(ms.x[:, e, :].T, (nrofp, 1))
        for o in range(ny):
            yo = np.zeros(N)
            for i in range(nu):
                yo += control.forced_response(g[o][i], T, Ue[:, i]).outputs
            if rng is not None and noise > 0:
                yo = yo + noise * rng.standard_normal(N)
            Y[:, o, e] = yo
        X[:, :, e] = Ue
    freq = ms.freq[ms.ex]
    true = np.array([[[complex(g[o][i](1j * 2 * np.pi * fk)) for fk in freq]
                      for i in range(nu)] for o in range(ny)])
    return g, ms, harm, X, Y, freq, true
