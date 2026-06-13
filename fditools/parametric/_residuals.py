"""Residual cost functions and the generalized SVD used by the parametric
estimators (port of ``mlfdi_res``/``nlsfdi_res``/``btlsfdi_res``/``fdicost``/
``qsvd``)."""

from __future__ import annotations

import numpy as np


def waxis_of(freq, cORd, fs=None):
    """Continuous (``s = jw``) or discrete (``z = e^{jw/fs}``) frequency axis."""
    freq = np.asarray(freq, dtype=float).ravel()
    if cORd == "d":
        return np.exp(1j * 2.0 * np.pi * freq / fs)
    return 1j * 2.0 * np.pi * freq


def _vander(waxis, hi, lo):
    """Vandermonde matrix with columns ``waxis**hi ... waxis**lo`` (descending)."""
    powers = np.arange(hi, lo - 1, -1)
    return waxis[:, None] ** powers[None, :]


def mlfdi_res(Bn, An, freq, X, Y, sX2, sY2, cXY, waxis):
    """Maximum-likelihood residual cost (port of ``mlfdi_res``)."""
    Bn = np.atleast_2d(np.asarray(Bn, dtype=float))
    An = np.asarray(An, dtype=float).ravel()
    n = An.size - 1
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.shape[0] == 1 and X.shape[1] != freq.size:
        X = X.T
    if Y.shape[0] == 1 and Y.shape[1] != freq.size:
        Y = Y.T
    nrofi, nrofo = X.shape[1], Y.shape[1]
    nrofh = nrofi * nrofo
    nroff = freq.size

    P = _vander(waxis, n, 0)
    Num = P @ Bn.T          # (nroff, nrofh)
    Den = P @ An            # (nroff,)

    E = []
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        SE = np.sqrt(sX2[:, i] * np.abs(Num[:, h]) ** 2
                     + sY2[:, o] * np.abs(Den) ** 2
                     - 2.0 * np.real(cXY[:, h] * Den * np.conj(Num[:, h])))
        E.append((Num[:, h] * X[:, i] - Den * Y[:, o]) / SE)
    E = np.concatenate(E)
    return (np.linalg.norm(E) ** 2) / 2.0


def nlsfdi_res(Bn, An, freq, FRF, FRF_W, cORd, fs=None):
    """Non-linear least-squares residual cost (port of ``nlsfdi_res``)."""
    Bn = np.atleast_2d(np.asarray(Bn, dtype=float))
    An = np.asarray(An, dtype=float).ravel()
    FRF = np.atleast_2d(FRF)
    FRF_W = np.atleast_2d(FRF_W)
    if FRF.shape[0] != freq.size:
        FRF = FRF.T
    if FRF_W.shape[0] != freq.size:
        FRF_W = FRF_W.T
    n = An.size - 1
    nrofh = Bn.shape[0]
    nrofb = Bn.shape[1]

    waxis = waxis_of(freq, cORd, fs)
    P = _vander(waxis, n, 0)
    Q = _vander(waxis, n, n - nrofb + 1)

    E = []
    for h in range(nrofh):
        E.append((FRF[:, h] - (Q @ Bn[h, :]) / (P @ An)) * FRF_W[:, h])
    E = np.concatenate(E)
    return (np.linalg.norm(E) ** 2) / 2.0


def btlsfdi_res(Bn, An, freq, X, Y, sX2, sY2, cXY, relax, waxis):
    """Bootstrapped-TLS residual cost (port of ``btlsfdi_res``)."""
    Bn = np.atleast_2d(np.asarray(Bn, dtype=float))
    An = np.asarray(An, dtype=float).ravel()
    n = An.size - 1
    nrofi, nrofo = X.shape[1], Y.shape[1]
    nrofh = nrofi * nrofo

    P = _vander(waxis, n, 0)
    Num = P @ Bn.T
    Den = P @ An

    E_all = []
    SE2_all = []
    SEr2_all = []
    for h in range(nrofh):
        i = h // nrofo
        o = h - i * nrofo
        base = (sX2[:, i] * np.abs(Num[:, h]) ** 2
                + sY2[:, o] * np.abs(Den) ** 2
                - 2.0 * np.real(cXY[:, h] * Den * np.conj(Num[:, h])))
        SEr2_all.append(base ** relax)
        SE2_all.append(base)
        E_all.append(np.abs(Num[:, h] * X[:, i] - Den * Y[:, o]) ** 2)
    E = np.concatenate(E_all)
    SE2 = np.concatenate(SE2_all)
    SEr2 = np.concatenate(SEr2_all)
    return np.sum(E / SEr2) / np.sum(SE2 / SEr2)


def fdicost(Bn, An, freq, X, Y, sX2, sY2, cXY, relax):
    """Residual cost used by ``costtest`` (continuous-time, single channel)."""
    freq = np.asarray(freq, dtype=float).ravel()
    waxis = 1j * 2.0 * np.pi * freq
    col = lambda a: np.asarray(a).reshape(freq.size, 1)
    return btlsfdi_res(Bn, An, freq, col(X), col(Y), col(sX2), col(sY2),
                       col(cXY), relax, waxis)


def qsvd(A, B):
    """Generalized (quotient) SVD — returns the right factor ``X`` only.

    ``A = Ua*Sa*X'`` , ``B = Ub*Sb*X'``.  Only ``X`` is needed by the GTLS/BTLS
    estimators (port of ``qsvd.m``).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    m, n = A.shape
    M = np.vstack([A, B])
    U, s, Vh = np.linalg.svd(M, full_matrices=True)
    V = Vh.conj().T
    S_top = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(S_top, s[:n])

    U1 = U[:m, :n]
    _, _, V1h = np.linalg.svd(U1, full_matrices=True)
    V1 = V1h.conj().T
    return V @ S_top @ V1
