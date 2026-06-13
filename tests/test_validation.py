import numpy as np
import pytest

import fditools as fdi
from conftest import freqresp_on


def _build_models(Pest):
    """A correct 2nd-order model and a deliberately wrong 1st-order model."""
    good, _ = fdi.mlfdi(Pest, 2, 0, 0, 200, 1e-10, 0, "c")
    bad, _ = fdi.lsfdi(Pest.userdata.X, Pest.userdata.Y, Pest.freq, 1, 0, 0, "c",
                       Pest.userdata.ms.harm.fs)
    return {"good": good, "bad": bad}


def test_chi2test_ranks_bad_model_worst(measurement):
    P0, ms, Pest = measurement
    SYS = _build_models(Pest)
    FRF = Pest.frf_columns()
    err, var, tag = fdi.chi2test(Pest.userdata.X, Pest.userdata.Y, Pest.freq,
                                 FRF, Pest.userdata.sCR, SYS)
    assert tag[0][0] == "bad"            # worst model listed first


def test_costtest_runs_and_orders(measurement):
    P0, ms, Pest = measurement
    SYS = _build_models(Pest)
    ud = Pest.userdata
    cost, interval, tag = fdi.costtest(ud.X, ud.Y, Pest.freq, ud.sX2, ud.sY2,
                                       ud.cXY, SYS, relax=1.0, nrofp=6)
    assert cost.shape[0] == 2
    assert interval[0] < interval[1]
    assert tag[0] == "bad"


def test_residtest_runs(measurement):
    P0, ms, Pest = measurement
    SYS = _build_models(Pest)
    FRF = Pest.frf_columns()
    # residtest scaling needs several periods of time data (nrofp > 3)
    x = np.tile(np.squeeze(ms.x[0, 0, :]), 6)
    out = fdi.residtest(x, x, Pest.freq, FRF, SYS, Pest.userdata.sCR, ms.harm.fs)
    lags, corr, cb50, frac50, tag, cb95, frac95 = out
    assert corr.shape[1] == 2
    assert lags.size == 2 * Pest.freq.size - 1
