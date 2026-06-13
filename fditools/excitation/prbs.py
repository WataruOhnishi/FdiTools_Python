"""Pseudo-random binary sequence (maximum-length) generation (port of ``prbs.m``)."""

from __future__ import annotations

import numpy as np

from ..auxiliary.misc import t2f

# maximum-length LFSR feedback taps (1-based register positions)
_TAPS = {
    2: [1, 2], 3: [2, 3], 4: [3, 4], 5: [3, 5], 6: [5, 6], 7: [4, 7],
    8: [4, 5, 6, 8], 9: [5, 9], 10: [7, 10], 11: [9, 11], 12: [6, 8, 11, 12],
    13: [9, 10, 12, 13], 14: [4, 8, 13, 14], 15: [14, 15], 16: [4, 13, 15, 16],
    17: [14, 17], 18: [11, 18], 19: [14, 17, 18, 19], 20: [17, 20], 21: [19, 21],
    22: [21, 22], 23: [18, 23], 24: [17, 22, 23, 24], 25: [22, 25],
    26: [20, 24, 25, 26], 27: [22, 25, 26, 27], 28: [25, 28], 29: [27, 29],
    30: [7, 28, 29, 30],
}


def prbs(fs, log2N, bitno=None, startnum=None):
    """Generate a PRBS / maximum-length binary sequence.

    Returns ``(bitseries, X, freq, nextstnum)`` with ``bitseries`` in ``{+1,-1}``.
    """
    if bitno is None:
        bitno = 2 ** log2N - 1
    if startnum is None:
        startnum = 2 ** log2N - 1

    startnum = int(round((abs(startnum - 1)) % (2 ** log2N - 1))) + 1
    stn = startnum
    reg = np.zeros(log2N)
    for i in range(log2N):
        reg[i] = stn % 2
        stn = (stn - reg[i]) / 2
    reg[reg == 0] = -1

    taps = np.array(_TAPS[log2N]) - 1  # 0-based
    bitseries = np.zeros(bitno)
    for i in range(bitno):
        bitseries[i] = -np.prod(reg[taps])
        reg = np.concatenate(([bitseries[i]], reg[:log2N - 1]))
    nextstnum = np.sum((reg / 2 + 0.5) * (2.0 ** np.arange(log2N)))

    X = t2f(bitseries, bitseries.size)
    freq = fs * np.arange(bitseries.size // 2) / bitseries.size
    return bitseries, X, freq, nextstnum
