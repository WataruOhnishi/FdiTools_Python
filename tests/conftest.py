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
