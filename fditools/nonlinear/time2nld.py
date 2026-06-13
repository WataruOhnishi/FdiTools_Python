"""Even/odd non-linear distortion detection from odd-odd multisine data
(port of ``time2nld.m``)."""

from __future__ import annotations

import numpy as np


def time2nld(x, y, fs, fl, fh, df):
    """Return ``(Yl, freql, Yo, freqo, Ye, freqe, Yn, freqn)``.

    Splits the output spectrum into linear, odd-nonlinear, even-nonlinear and
    underlying-noise contributions.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    nrofi = x.shape[1]
    nrofo = y.shape[1]
    nrofs = int(round(fs / df))
    nl = int(np.ceil(fl / df))
    nh = int(np.floor(fh / df))
    freq = np.arange(nl, nh + 1) * df
    nroff = freq.size
    nrofp = x.shape[0] // nrofs

    Xs = np.zeros((nroff, nrofi), dtype=complex)
    Ys = np.zeros((nroff, nrofo), dtype=complex)
    for i in range(nrofi):
        acc = np.zeros((nroff, nrofp), dtype=complex)
        for p in range(nrofp):
            Ip = np.fft.fft(x[p * nrofs:(p + 1) * nrofs, i])
            acc[:, p] = Ip[nl:nh + 1]
        Xs[:, i] = acc.mean(axis=1)
    for o in range(nrofo):
        acc = np.zeros((nroff, nrofp), dtype=complex)
        for p in range(nrofp):
            Op = np.fft.fft(y[p * nrofs:(p + 1) * nrofs, o])
            acc[:, p] = Op[nl:nh + 1]
        Ys[:, o] = acc.mean(axis=1)

    # underlying noise from 2-period blocks (in-between lines)
    nph = nrofp // 2
    Yn = np.zeros((nroff, nrofo), dtype=complex)
    for o in range(nrofo):
        NSE = np.zeros((nroff, nph), dtype=complex)
        for p in range(nph):
            Op = np.fft.fft(y[p * nrofs * 2:(p + 1) * nrofs * 2, o])
            block = Op[2 * nl - 1:2 * nh + 1]
            NSE[:, p] = block[::2]
        Yn[:, o] = NSE.mean(axis=1)
    freqn = freq

    # initial frequency phase from the strongest of the first 4 input lines
    s = int(np.argmax(np.abs(Xs[:4, 0]))) + 1     # 1-based, 1..4
    even, odd = {1: (1, 2), 2: (-1, 2), 3: (-1, -2), 4: (-3, -2)}[s]

    freql, Yl = [], []
    freqe, Ye = [], []
    freqo, Yo = [], []
    for f in range(s - 1, nroff, 4):
        freql.append(freq[f])
        Yl.append(Ys[f, :])
    for f in range(s + even - 1, nroff, 2):
        freqe.append(freq[f])
        Ye.append(Ys[f, :])
    for f in range(s + odd - 1, nroff, 4):
        freqo.append(freq[f])
        Yo.append(Ys[f, :])

    return (np.array(Yl), np.array(freql),
            np.array(Yo), np.array(freqo),
            np.array(Ye), np.array(freqe),
            Yn, freqn)
