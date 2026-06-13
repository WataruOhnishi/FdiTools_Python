"""Private helpers for excitation design (port of the ``private`` folder:
``effval``/``lin2qlog``/``lpnorm``/``orthogonal``/``randph``/``schroed``/
``msinl2p``).

Frequency-line index sets (``Fe``/``Fa``/``Ft``) are kept **0-based** at the
Python API boundary; internally :func:`msinl2p` follows the MATLAB algebra.
"""

from __future__ import annotations

import numpy as np

from ..auxiliary.misc import f2t


def effval(X, Fe=None):
    """Effective (RMS) value of a spectrum with DC correction.

    Port of ``effval.m``: the DC bin is scaled by ``sqrt(2)`` before the
    optional restriction to the harmonic set *Fe* (0-based indices).
    """
    X = np.asarray(X, dtype=complex).ravel().copy()
    X[0] = np.sqrt(2.0) * X[0]
    if Fe is not None:
        X = X[np.asarray(Fe, dtype=int)]
    return np.sqrt(np.sum(np.abs(X) ** 2) / 2.0)


def lpnorm(A, p=2):
    """Vector/column ``Lp`` norm normalised by ``nRow**(1/p)`` (port of ``lpnorm``)."""
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
        squeeze = True
    else:
        squeeze = False
    nrow = A.shape[0]
    if np.isinf(p):
        Lp = np.max(np.abs(A), axis=0)
    else:
        Lp = np.sum(np.abs(A) ** p, axis=0) ** (1.0 / p) / (nrow ** (1.0 / p))
    return Lp[0] if squeeze else Lp


def lin2qlog(flin, fr):
    """Linear -> quasi-logarithmic frequency-line grid (port of ``lin2qlog``).

    Returns ``(fqlog, fidx)`` where *fqlog* are the kept line numbers and
    *fidx* their 0-based positions in *flin*.
    """
    flin = np.asarray(flin)
    n = flin.size
    last = 0
    fidx = [0]
    fqlog = [flin[0]]
    while last < n - 1:
        cand = flin[last + 1:]
        nxt = int(np.argmin(np.abs(flin[last] * fr - cand)))
        last = last + 1 + nxt
        if flin[last] > fqlog[-1] * np.sqrt(fr):
            fidx.append(last)
            fqlog.append(flin[last])
    return np.array(fqlog), np.array(fidx, dtype=int)


def orthogonal(nrofi):
    """Orthogonal-multisine transformation matrix (port of ``orthogonal``)."""
    p = np.arange(nrofi)[:, None]
    q = np.arange(nrofi)[None, :]
    return nrofi ** (-0.5) * np.exp(1j * 2.0 * np.pi * p * q / nrofi)


def randph(X, rng_seed):
    """Random phases in ``[-pi, pi]`` keeping ``abs(X)`` (port of ``randph``).

    Uses NumPy's Mersenne-Twister; the *sequence* differs from MATLAB's ``rng``
    so designs are reproducible within Python but not bit-identical to MATLAB.
    """
    X = np.asarray(X, dtype=complex)
    ampl = np.abs(X)
    rng = np.random.RandomState(int(rng_seed))
    phase = (2.0 * rng.random_sample(ampl.shape) - 1.0) * np.pi
    return ampl * np.exp(1j * phase)


def schroed(X):
    """Schroeder phases (port of ``schroed``)."""
    X = np.asarray(X, dtype=complex)
    one_d = X.ndim == 1
    if one_d:
        X = X[:, None]
    ampl = np.abs(X)
    freqno = ampl.shape[0]
    phase = np.zeros_like(ampl)
    amplnorm = ampl / np.sqrt(np.sum(ampl ** 2, axis=0, keepdims=True))
    amplnorm = 2.0 * np.pi * amplnorm ** 2
    for i in range(2, freqno):
        phase[i, :] = phase[i - 1, :] - np.sum(amplnorm[:i, :], axis=0)
    out = ampl * np.exp(1j * phase)
    return out[:, 0] if one_d else out


def msinl2p(X, nrofs, itp, Fa=None, W=0, H=None):
    """L2p-norm optimisation of multisine phases (port of ``msinl2p.m``).

    Only the in-only minimisation (``Fa`` empty, ``H`` empty) – the path used by
    :func:`fditools.excitation.multisine` – is implemented; the additional-
    harmonic (snow) and input/output weighting branches raise
    ``NotImplementedError``.
    """
    if Fa is not None and len(np.atleast_1d(Fa)):
        raise NotImplementedError("msinl2p: additional-harmonic (Fa) branch not ported")
    if H is not None:
        raise NotImplementedError("msinl2p: input/output (H) branch not ported")

    X = np.asarray(X, dtype=complex).ravel().copy()
    p = 2
    Fe = np.nonzero(X)[0]              # 0-based effective lines
    Ft = Fe.copy()
    nrofe = Fe.size
    kmax = int(np.max(Ft))            # largest harmonic number (== 1-based max - 1)
    if X.size <= kmax:
        X = np.concatenate([X, np.zeros(kmax + 1 - X.size, dtype=complex)])

    if itp in ("r", "random"):
        relvar, iterno = 1e-6, 10
    else:
        relvar, iterno = 1e-10, 20

    CF = np.nan
    while p < 500:
        # match L2p summation with integration criterion
        Ndummy = 2 * p * kmax + 1
        N = 2
        while Ndummy >= N:
            N *= 2
        N = min(N, nrofs)

        W0 = 1.0 / effval(X, Fe)
        X = X * W0
        x = f2t(X, N)
        cost = lpnorm(x.ravel(), 2 * p)

        relax = 0.01
        it = 0
        relerror = np.inf
        X0 = X.copy()
        cost0 = cost

        # (k-l)+N and (k+l)+N index matrices into the 1-based X2p2 array
        ft1 = Ft + 1                                   # 1-based harmonic indices
        idx_mins = (ft1[:, None] - ft1[None, :]) + N   # Ft(k)-Ft(l)+N
        idx_plus = (ft1[:, None] + ft1[None, :] - 2) + N  # (Ft(k)-1)+(Ft(l)-1)+N
        phkphlmins = idx_mins.flatten(order="F") - 1   # -> 0-based python index
        phkphlplus = idx_plus.flatten(order="F") - 1

        while it < iterno and relerror > relvar:
            it += 1
            X2p2 = np.fft.fft(x ** (2 * p - 2))
            X2p2 = np.concatenate([np.conj(X2p2[1:N])[::-1], X2p2])  # length 2N-1

            XFt = X[Ft]
            Qmins = np.conj(np.outer(XFt, np.conj(XFt)))   # conj(X(Ft)*X(Ft)')
            Qplus = np.conj(np.outer(XFt, XFt))            # conj(X(Ft)*X(Ft).')
            Qmins = Qmins.flatten(order="F") * X2p2[phkphlmins]
            Qplus = Qplus.flatten(order="F") * X2p2[phkphlplus]
            Qmins = Qmins.reshape((nrofe, nrofe), order="F")
            Qplus = Qplus.reshape((nrofe, nrofe), order="F")

            JphJph = p * np.real(Qmins - Qplus)
            JphE = np.sum(np.imag(Qmins + Qplus), axis=1)

            A = JphJph
            d = np.diag(A)
            A = A + relax * np.diag(d + np.max(d) * np.finfo(float).eps)
            Delta = -np.linalg.solve(A, JphE)

            X[Fe] = np.abs(X[Fe]) * np.exp(1j * (np.angle(X[Fe]) + Delta[:nrofe]))
            x = f2t(X, N)
            cost = lpnorm(x.ravel(), 2 * p)
            relerror = abs(cost - cost0) / cost0
            if cost < cost0:
                relax /= 2.0
                X0 = X.copy()
                cost0 = cost
            else:
                relax *= 10.0
                X = X0.copy()
                cost = cost0
                x = f2t(X, N)
            CF = lpnorm(x.ravel(), np.inf) / effval(X, Fe)

        X = X0 / W0
        p = int(np.ceil(p * 2))

    return X
