"""Local Polynomial Method (LPM) FRF estimation (port of ``time2frf_lpm.m``).

The LPM estimates the FRF together with the transient (leakage) term, giving
low-bias FRF estimates even with a long start-up transient (no transient period
need be discarded).  SISO / SIMO (single input).

* **periodic** mode (``period=N, lines=ex``): a single ``P*N``-point DFT of the
  whole ``P``-period record; the bins between excited lines carry the transient,
  fitted by a local polynomial of order ``R`` over ``2*halfwidth+1`` neighbours.
* **broadband** mode (default): consecutive-bin local polynomial over the whole
  record DFT, for arbitrary / non-periodic excitation.

Reference: Pintelon, Barbe, Vandersteen, Schoukens, MSSP 25(7) 2683-2704 (2011);
Pintelon & Schoukens 2012, Ch. 7.
"""

from __future__ import annotations

import numpy as np

from ..frfdata import FrfData, UserData


def _lpm_mimo_orthogonal(u, y, fs, R, n, band, nrofs, ex):
    """Orthogonal multi-experiment MIMO LPM (transient-free spike + solve).

    u : (N, nu, ne), y : (N, ny, ne) ; ex : 0-based excited bins.
    Returns an ``(ny, nu, nl)`` :class:`FrfData`.
    """
    nu, ne = u.shape[1], u.shape[2]
    ny = y.shape[1]
    df = fs / nrofs
    freq = ex * df
    nl = ex.size
    powers = np.arange(R + 1)

    Y0 = np.zeros((ny, ne, nl), dtype=complex)
    U0 = np.zeros((nu, ne, nl), dtype=complex)
    vY0 = np.zeros((ny, ne, nl))
    for e in range(ne):
        P = u.shape[0] // nrofs
        if P < 2:
            raise ValueError("orthogonal LPM needs >= 2 periods.")
        nn = n if n < P else P - 1
        nw = 2 * nn + 1
        nprm = R + 2
        if nw < nprm:
            raise ValueError(f"window too small: 2*halfwidth+1 (={nw}) < "
                             f"order+2 (={nprm}).")
        PN = P * nrofs
        Ye = np.fft.fft(y[:PN, :, e], axis=0)
        Ue = np.fft.fft(u[:PN, :, e], axis=0)
        Kidx = P * ex                                  # 0-based bins in PN grid
        m = np.arange(-nn, nn + 1)
        Kr = np.hstack([(m == 0).astype(float)[:, None],
                        (m[:, None].astype(float)) ** powers[None, :]])
        c11 = np.real(np.diag(np.linalg.inv(Kr.T @ Kr)))[0]
        q = max(nw - nprm, 1)
        for j in range(nl):
            rr = np.clip(Kidx[j] + m, 0, PN - 1)
            Yw = Ye[rr, :]
            th, *_ = np.linalg.lstsq(Kr, Yw, rcond=None)
            Y0[:, e, j] = th[0, :]
            U0[:, e, j] = Ue[Kidx[j], :]
            res = Yw - Kr @ th
            vY0[:, e, j] = np.real(np.sum(np.conj(res) * res, axis=0)) / q * c11

    G = np.zeros((ny, nu, nl), dtype=complex)
    sG = np.zeros((ny, nu, nl))
    for j in range(nl):
        Um = U0[:, :, j]
        Ym = Y0[:, :, j]
        W = np.linalg.pinv(Um)
        G[:, :, j] = Ym @ W
        for i in range(nu):
            sG[:, i, j] = np.sqrt(vY0[:, :, j] @ np.abs(W[:, i]) ** 2)

    if band is not None:
        sel = (freq >= band[0]) & (freq <= band[1])
        freq, G, sG = freq[sel], G[:, :, sel], sG[:, :, sel]
    ud = UserData(sG=sG, method="lpm", nrofp=ne)
    return FrfData(G, freq, userdata=ud)


def _lpm_mimo_zippered(u, y, fs, R, n, band, nrofs, ex):
    """Single zippered MIMO experiment: per-input SIMO periodic LPM + assemble.

    Each input owns disjoint (interleaved) excited lines; the active input is
    identified per line, the SIMO LPM is run on its owned lines, and every
    column is interpolated onto the full grid (per-channel resolution = 1/nu).
    Useful e.g. for thermal systems (one experiment, several inputs, long
    transient handled by the LPM).  Returns an ``(ny, nu, nl)`` :class:`FrfData`.
    """
    nu = u.shape[1]
    ny = y.shape[1]
    df = fs / nrofs
    freq = ex * df
    nl = ex.size
    P = u.shape[0] // nrofs

    Uavg = np.zeros((nl, nu), dtype=complex)
    for i in range(nu):
        acc = np.zeros(nrofs, dtype=complex)
        for p in range(P):
            acc += np.fft.fft(u[p * nrofs:(p + 1) * nrofs, i])
        Uavg[:, i] = acc[ex] / P
    owner = np.argmax(np.abs(Uavg), axis=1)            # input owning each line

    G = np.full((ny, nu, nl), np.nan, dtype=complex)
    sGm = np.full((ny, nu, nl), np.nan)
    Tm = np.full((ny, nu, nl), np.nan, dtype=complex)
    for i in range(nu):
        idx = np.where(owner == i)[0]
        if idx.size == 0:
            continue
        FRFi, _, sGi, Ti = time2frf_lpm(u[:, i], y, fs, order=R, halfwidth=n,
                                        period=nrofs, lines=ex[idx])
        for o in range(ny):
            G[o, i, idx] = FRFi[:, o]
            sGm[o, i, idx] = sGi[:, o]
            Tm[o, i, idx] = Ti[:, o]

    for i in range(nu):                                # interpolate onto full grid
        fi = np.where(~np.isnan(G[0, i, :]))[0]
        for o in range(ny):
            G[o, i, :] = (np.interp(freq, freq[fi], np.real(G[o, i, fi]))
                          + 1j * np.interp(freq, freq[fi], np.imag(G[o, i, fi])))
            sGm[o, i, :] = np.interp(freq, freq[fi], sGm[o, i, fi])

    if band is not None:
        sel = (freq >= band[0]) & (freq <= band[1])
        freq, G, sGm, Tm = freq[sel], G[:, :, sel], sGm[:, :, sel], Tm[:, :, sel]
    ud = UserData(sG=sGm, T=Tm, method="lpm")
    return FrfData(G, freq, userdata=ud)


def _lpm_core(U, Y, R, n):
    """Broadband consecutive-bin LPM over grid 0..L-1."""
    L = U.shape[0]
    nrofo = Y.shape[1]
    npar = 2 * (R + 1)
    powers = np.arange(R + 1)
    G = np.zeros((L, nrofo), dtype=complex)
    T = np.zeros((L, nrofo), dtype=complex)
    sG = np.zeros((L, nrofo))
    for m in range(L):
        lo, hi = m - n, m + n
        if lo < 0:
            lo, hi = 0, min(2 * n, L - 1)
        if hi > L - 1:
            hi, lo = L - 1, max(0, L - 1 - 2 * n)
        idx = np.arange(lo, hi + 1)
        r = (idx - m).astype(float)
        Rp = r[:, None] ** powers[None, :]              # (nw, R+1)
        K = np.hstack([U[idx][:, None] * Rp, Rp])       # (nw, 2(R+1))
        Yw = Y[idx, :]
        th, *_ = np.linalg.lstsq(K, Yw, rcond=None)
        G[m, :] = th[0, :]
        T[m, :] = th[R + 1, :]
        q = idx.size - npar
        if q >= 1:
            c11 = np.real(np.diag(np.linalg.inv(K.conj().T @ K)))[0]
            res = Yw - K @ th
            s2 = np.real(np.sum(np.conj(res) * res, axis=0)) / q
            sG[m, :] = np.sqrt(np.abs(s2 * c11))   # abs guards edge ill-conditioning
        else:
            sG[m, :] = np.nan
    return G, T, sG


def time2frf_lpm(u, y, fs, order=2, halfwidth=2, band=None,
                 period=None, lines=None):
    """Return ``(FRF, freq, sG, T)``.

    Parameters
    ----------
    u : (N,) or (N, 1)      single-input time data
    y : (N,) or (N, nrofo)  output time data
    fs : float              sampling frequency [Hz]
    order : int             transient polynomial order ``R`` (default 2)
    halfwidth : int         half window ``n`` (neighbours each side; default 2)
    band : (fl, fh), optional   frequency band to return (default: full)
    period : int, optional      samples per period ``N`` -> enables periodic mode
    lines : array_like, optional excited bin indices (**0-based**, into 0..N/2),
        i.e. ``freq = lines * fs / period`` (periodic mode)

    Returns
    -------
    SISO/SIMO : ``(FRF, freq, sG, T)`` arrays.
    MIMO (``u`` of shape ``(N, nu, ne)`` with ``period``/``lines``):
        an ``(ny, nu, nl)`` :class:`FrfData` from the orthogonal
        multiple-experiment solve.
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if u.ndim == 3:                                   # orthogonal MIMO LPM
        if period is None or lines is None:
            raise ValueError("MIMO LPM needs 'period' and 'lines'.")
        return _lpm_mimo_orthogonal(u, y, fs, int(order), int(halfwidth), band,
                                    int(period), np.asarray(lines, dtype=int).ravel())
    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have equal length.")
    if u.shape[1] > 1:                                # single zippered MIMO exp.
        if period is None or lines is None:
            raise ValueError("zippered MIMO LPM needs 'period' and 'lines'.")
        return _lpm_mimo_zippered(u, y, fs, int(order), int(halfwidth), band,
                                  int(period), np.asarray(lines, dtype=int).ravel())
    Ntot, nrofi = u.shape
    R, n = int(order), int(halfwidth)
    periodic = period is not None and lines is not None

    if periodic:
        Nper = int(period)
        P = Ntot // Nper
        if P < 2:
            raise ValueError("periodic LPM needs >= 2 periods.")
        PN = P * Nper
        U = np.fft.fft(u[:PN, 0])
        Y = np.fft.fft(y[:PN, :], axis=0)
        exb = np.asarray(lines, dtype=int).ravel()       # 0-based 1-period bins
        Kidx = P * exb                                   # 0-based bins in PN grid
        freq_all = exb * fs / Nper
        nl = exb.size
        if n >= P:
            n = P - 1
        nprm = R + 2
        nw = 2 * n + 1
        if nw < nprm:
            raise ValueError(f"window too small: 2*halfwidth+1 (={nw}) < "
                             f"order+2 (={nprm}).")
        powers = np.arange(R + 1)
        nrofo = Y.shape[1]
        FRF = np.zeros((nl, nrofo), dtype=complex)
        T = np.zeros((nl, nrofo), dtype=complex)
        sG = np.zeros((nl, nrofo))
        m = np.arange(-n, n + 1)
        for idx in range(nl):
            K = Kidx[idx]
            rr = np.clip(K + m, 0, PN - 1)
            Kr = np.hstack([(m == 0).astype(float)[:, None],
                            (m[:, None].astype(float)) ** powers[None, :]])
            Yw = Y[rr, :]
            th, *_ = np.linalg.lstsq(Kr, Yw, rcond=None)
            FRF[idx, :] = th[0, :] / U[K]
            T[idx, :] = th[1, :]
            q = nw - nprm
            if q >= 1:
                c11 = np.real(np.diag(np.linalg.inv(Kr.T @ Kr)))[0]
                res = Yw - Kr @ th
                s2 = np.real(np.sum(np.conj(res) * res, axis=0)) / q
                sG[idx, :] = np.sqrt(s2 * c11) / np.abs(U[K])
            else:
                sG[idx, :] = np.nan
    else:
        Mmax = Ntot // 2
        U = np.fft.fft(u[:, 0])
        Y = np.fft.fft(y, axis=0)
        Uvec = U[1:Mmax + 1]
        Yvec = Y[1:Mmax + 1, :]
        freq_all = np.arange(1, Mmax + 1) * fs / Ntot
        nprm = 2 * (R + 1)
        if 2 * n + 1 <= nprm:
            raise ValueError("window too small: need 2*halfwidth+1 > 2*(order+1).")
        FRF, T, sG = _lpm_core(Uvec, Yvec, R, n)

    if band is None:
        sel = np.ones(freq_all.shape, dtype=bool)
    else:
        sel = (freq_all >= band[0]) & (freq_all <= band[1])
    return FRF[sel, :], freq_all[sel], sG[sel, :], T[sel, :]
