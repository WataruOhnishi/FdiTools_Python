"""Frequency-domain subspace identification of a discrete state-space model
(McKelvey's method) — port of ``ssfdi.m`` (WORK IN PROGRESS in the original).

The interactive order prompt of the MATLAB code is replaced by a required
*order* argument.
"""

from __future__ import annotations

import numpy as np


def ssfdi(FRF, FRF_W, freq, n_in, n_out, n_row, n_col, order):
    """Return ``(A, B, C, D, fs)`` of a discrete-time state-space model.

    FRF, FRF_W : (nrofmea, n_in*n_out) column-stacked FRF / weighting
        Column order ``[H11 H21 H31 ... H12 H22 ...]``.
    freq : (nrofmea,) frequency axis; ``max(freq)`` becomes the Nyquist freq.
    """
    FRF = np.atleast_2d(np.asarray(FRF, dtype=complex))
    FRF_W = np.atleast_2d(np.asarray(FRF_W, dtype=complex))
    freq = np.asarray(freq, dtype=float).ravel()
    nr = freq.size
    fs = 2.0 * np.max(freq)

    # mirror to a full periodic spectrum
    FRF = np.vstack([FRF, np.conj(FRF[nr - 2:0:-1, :])])
    FRF[nr - 1, :] = np.real(FRF[nr - 1, :])
    N2 = 2 * nr - 2
    omega = 2.0 * np.pi * np.arange(N2) / N2
    ifrf = np.real(np.fft.ifft(FRF, axis=0))

    # block-Hankel matrix from the impulse response
    H = np.column_stack([
        ifrf[:, n_out * i:n_out * (i + 1)].T.flatten(order="F")
        for i in range(n_in)
    ])  # (n_out*N2, n_in)
    Hqr = np.hstack([
        H[n_out * (i + 1):n_out * (i + 1 + n_row), :]
        for i in range(n_col)
    ])  # (n_out*n_row, n_in*n_col)

    UU, ss, VVh = np.linalg.svd(Hqr, full_matrices=True)
    n = order
    U1 = UU[:, :n]
    S1 = np.diag(ss[:n])
    Li = U1 @ np.sqrt(S1)

    C = Li[:n_out, :]
    A = np.linalg.pinv(Li[:n_out * (n_row - 1), :]) @ Li[n_out:n_out * n_row, :]

    # rebuild H / weighting on the (positive) measured frequencies
    Hc = np.column_stack([
        np.conj(FRF[:, n_out * i:n_out * (i + 1)]).T.flatten(order="F")
        for i in range(n_in)
    ])
    Wght = np.column_stack([
        np.conj(FRF_W[:, n_out * i:n_out * (i + 1)]).T.flatten(order="F")
        for i in range(n_in)
    ])

    rows = n_out * nr
    Sblk = []
    eyeo = np.eye(n_out)
    for i in range(nr):
        CA = C @ np.linalg.inv(np.exp(1j * omega[i]) * np.eye(n) - A)
        Sblk.append(np.hstack([eyeo, CA]))
    S = np.vstack(Sblk)
    S = np.vstack([np.real(S), np.imag(S)])

    D_cols = []
    B_cols = []
    for i in range(n_in):
        w = np.concatenate([Wght[:rows, i], Wght[:rows, i]])
        W = np.outer(w, np.ones(n_out + n))
        rhs = w * np.concatenate([np.real(Hc[:rows, i]), np.imag(Hc[:rows, i])])
        DB = np.linalg.pinv(W * S) @ rhs
        D_cols.append(DB[:n_out])
        B_cols.append(DB[n_out:n_out + n])
    D = np.column_stack(D_cols)
    B = np.column_stack(B_cols)
    return A, B, C, D, fs
