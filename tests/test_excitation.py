import numpy as np
import control
import pytest

import fditools as fdi
from fditools.auxiliary.misc import f2t, t2f


def test_multisine_basic_shapes_and_real():
    harm = dict(fs=1000.0, df=1.0, fl=2.0, fh=200.0, fr=1.02)
    opt = dict(itp="r", ctp="n", dtp="f", gtp="l")
    ms = fdi.multisine(harm, control.tf([1], [1]), opt)
    assert ms.x.shape == (1, 1, 1000)
    sig = np.squeeze(ms.x)
    assert np.isrealobj(sig) or np.allclose(sig.imag, 0)
    # one period is periodic by construction: rms normalised to ~1
    assert np.isclose(np.sqrt(np.mean(sig ** 2)), 1.0, atol=1e-6)
    # spectrum only on excited lines
    assert ms.ex.size > 0


def test_multisine_excites_requested_band():
    harm = dict(fs=2000.0, df=1.0, fl=10.0, fh=300.0, fr=1.02)
    opt = dict(itp="s", ctp="n", dtp="f", gtp="l")
    ms = fdi.multisine(harm, control.tf([1], [1]), opt)
    exfreq = ms.freq[ms.ex]
    assert exfreq.min() >= 10.0 - 1.0
    assert exfreq.max() <= 300.0 + 1.0


def test_multisine_compression_lowers_crest_factor():
    harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=200.0, fr=1.02)
    base = dict(itp="s", dtp="f", gtp="l")
    ms_n = fdi.multisine(harm, control.tf([1], [1]), dict(ctp="n", **base))
    ms_c = fdi.multisine(harm, control.tf([1], [1]), dict(ctp="c", **base))
    assert ms_c.cf[0, 0] <= ms_n.cf[0, 0] + 1e-6


def test_multisine_qlog_is_sparser_than_linear():
    harm = dict(fs=2000.0, df=1.0, fl=5.0, fh=400.0, fr=1.05)
    opt = dict(itp="r", ctp="n", dtp="f")
    ms_lin = fdi.multisine(harm, control.tf([1], [1]), dict(gtp="l", **opt))
    ms_q = fdi.multisine(harm, control.tf([1], [1]), dict(gtp="q", **opt))
    assert ms_q.ex.size < ms_lin.ex.size


def test_f2t_matches_definition():
    # f2t(X, N) == N * real(ifft(X, N))  (port of f2t.m)
    rng = np.random.default_rng(1)
    X = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    assert np.allclose(f2t(X, 64), 64 * np.real(np.fft.ifft(X, 64)))


def test_t2f_matches_definition():
    # t2f(x, N): single-sided coefficients, DC un-doubled (port of t2f.m)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(64)
    N = 64
    Xf = np.fft.fft(x, N)
    expected = np.concatenate([Xf[0:1], 2.0 * Xf[1:N // 2]]) / N
    assert np.allclose(t2f(x, N), expected)


def test_f2t_t2f_reconstruct_bandlimited_tone():
    # for a tone strictly inside the band, t2f then a manual full-spectrum
    # rebuild via f2t reproduces the signal
    N = 64
    n = np.arange(N)
    x = np.cos(2 * np.pi * 5 * n / N)
    X = t2f(x, N)                      # single-sided
    full = np.zeros(N, dtype=complex)
    full[0] = X[0]
    full[1:N // 2] = X[1:] / 2.0
    full[N // 2 + 1:] = np.conj(X[1:][::-1]) / 2.0
    xr = f2t(full, N)
    assert np.allclose(xr, x, atol=1e-10)


def test_sweptsine_runs():
    h = dict(fs=1000.0, df=1.0, fl=1.0, fh=400.0)
    for kind in ("lin", "qdr", "log"):
        out = fdi.sweptsine(h, dict(type=kind))
        assert out.x.size == 1000
        assert np.all(np.isfinite(out.x))


def test_prbs_properties():
    bitseries, X, freq, nextstnum = fdi.prbs(1000.0, 7)
    assert bitseries.size == 2 ** 7 - 1
    assert set(np.unique(bitseries)).issubset({-1.0, 1.0})


def test_multisine2hdr_writes_file(tmp_path):
    harm = dict(fs=1000.0, df=1.0, fl=2.0, fh=100.0, fr=1.02)
    opt = dict(itp="s", ctp="n", dtp="f", gtp="l")
    ms = fdi.multisine(harm, control.tf([1], [1]), opt)
    path = fdi.multisine2hdr(ms, str(tmp_path / "ms.h"))
    text = open(path).read()
    assert "NROFS_ms" in text and "refvec_ms" in text
