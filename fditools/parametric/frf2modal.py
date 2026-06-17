"""Structured modal identification of MIMO systems from FRF data
(port of ``frf2modal.m``).

Two-stage, structured (rank-one residue) approach of M. van der Hulst et al.,
"Structured identification of multivariable modal systems", MSSP 247 (2026)
113948.

* **Stage 1** - additive model: poles ``(wn, ze)`` by Levenberg-Marquardt,
  residue matrices by per-I/O linear least squares (variable projection).
* **rank-one SVD** initialisation of the mode shapes (Eckart-Young).
* **Stage 2** - refine the rank-one modal model against the FRF.

Two damping models: ``'proportional'`` (real mode shapes) and ``'general'``
(complex mode shapes, ``L_i = psi_l psi_r^T``).  The proportional path is
implemented with real mode shapes throughout (the correct structure for
proportional damping); the general path carries complex shapes via a real/imag
split, as in the MATLAB code.

Returns ``(modal, Pm)`` with *modal* a parameter ``dict`` and *Pm* a real
``control.StateSpace``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import control

from ..frfdata import FrfData


# --------------------------------------------------------------------------- #
# basis / residue helpers
# --------------------------------------------------------------------------- #
def _basis(xi, wn, ze, nrbm, hasD, gen):
    N = xi.size
    cols = []
    if hasD:
        cols.append(np.ones(N, dtype=complex))
    if nrbm > 0:
        cols.append(1.0 / xi ** 2)
    for i in range(wn.size):
        Ai = xi ** 2 + 2 * ze[i] * wn[i] * xi + wn[i] ** 2
        cols.append(1.0 / Ai)
        if gen:
            cols.append(xi / Ai)
    return np.column_stack(cols)


def _stage1_residual(p, G, xi, Wt, nrbm, nflex, hasD, gen, want_C=False):
    ny, nu, N = G.shape
    wn = p[:nflex]
    ze = p[nflex:]
    B = _basis(xi, wn, ze, nrbm, hasD, gen)
    Nb = B.shape[1]
    C = np.zeros((Nb, ny, nu), dtype=complex)
    parts = []
    for o in range(ny):
        for u in range(nu):
            g = G[o, u, :]
            sw = np.sqrt(Wt[o, u, :])
            Bw = B * sw[:, None]
            c, *_ = np.linalg.lstsq(Bw, g * sw, rcond=None)
            C[:, o, u] = c
            e = (g - B @ c) * sw
            parts.append(np.real(e))
            parts.append(np.imag(e))
    r = np.concatenate(parts)
    if want_C:
        return r, C
    return r


def _unpack_residues(C, ny, nu, nrbm, nflex, hasD, gen):
    D = np.zeros((ny, nu), dtype=complex)
    Rrbm = np.zeros((ny, nu), dtype=complex)
    B0 = np.zeros((ny, nu, nflex), dtype=complex)
    B1 = np.zeros((ny, nu, nflex), dtype=complex)
    row = 0
    if hasD:
        D = C[row]
        row += 1
    if nrbm > 0:
        Rrbm = C[row]
        row += 1
    for i in range(nflex):
        B0[:, :, i] = C[row]
        row += 1
        if gen:
            B1[:, :, i] = C[row]
            row += 1
    return D, Rrbm, B0, B1


# --------------------------------------------------------------------------- #
# modal parameter (un)packing for stage 2
# --------------------------------------------------------------------------- #
def _F(a):
    return np.asarray(a).ravel(order="F")


def _unpack_modal(rho, ny, nu, nrbm, nflex, hasD, gen):
    i0 = 0
    wn = rho[i0:i0 + nflex]; i0 += nflex
    ze = rho[i0:i0 + nflex]; i0 += nflex
    ms = {}
    if gen:
        rl = rho[i0:i0 + ny * nflex].reshape(ny, nflex, order="F"); i0 += ny * nflex
        il = rho[i0:i0 + ny * nflex].reshape(ny, nflex, order="F"); i0 += ny * nflex
        rr = rho[i0:i0 + nu * nflex].reshape(nu, nflex, order="F"); i0 += nu * nflex
        ir = rho[i0:i0 + nu * nflex].reshape(nu, nflex, order="F"); i0 += nu * nflex
        ms["psil"] = rl + 1j * il
        ms["psir"] = rr + 1j * ir
    else:
        ms["phil"] = rho[i0:i0 + ny * nflex].reshape(ny, nflex, order="F"); i0 += ny * nflex
        ms["phir"] = rho[i0:i0 + nu * nflex].reshape(nu, nflex, order="F"); i0 += nu * nflex
    phil_r = rho[i0:i0 + ny * nrbm].reshape(ny, nrbm, order="F"); i0 += ny * nrbm
    phir_r = rho[i0:i0 + nu * nrbm].reshape(nu, nrbm, order="F"); i0 += nu * nrbm
    if hasD:
        D = rho[i0:i0 + ny * nu].reshape(ny, nu, order="F")
    else:
        D = np.zeros((ny, nu))
    return wn, ze, ms, phil_r, phir_r, D


def _stage2_residual(rho, G, xi, Wt, ny, nu, nrbm, nflex, hasD, gen):
    wn, ze, ms, phil_r, phir_r, D = _unpack_modal(rho, ny, nu, nrbm, nflex, hasD, gen)
    N = xi.size
    P = np.broadcast_to(D[:, :, None].astype(complex), (ny, nu, N)).copy()
    for j in range(nrbm):
        Rj = np.outer(phil_r[:, j], phir_r[:, j])
        P += Rj[:, :, None] * (1.0 / xi ** 2)[None, None, :]
    for i in range(nflex):
        w, z = wn[i], ze[i]
        if gen:
            lam = -z * w + 1j * w * np.sqrt(max(1 - z ** 2, 0.0))
            Li = np.outer(ms["psil"][:, i], ms["psir"][:, i])
            di = 1.0 / (xi - lam)
            dc = 1.0 / (xi - np.conj(lam))
            P += Li[:, :, None] * di[None, None, :] \
                + np.conj(Li)[:, :, None] * dc[None, None, :]
        else:
            Ri = np.outer(ms["phil"][:, i], ms["phir"][:, i])
            di = 1.0 / (xi ** 2 + 2 * z * w * xi + w ** 2)
            P += Ri[:, :, None] * di[None, None, :]
    E = (G - P) * np.sqrt(Wt)
    return np.concatenate([np.real(E).ravel(), np.imag(E).ravel()])


# --------------------------------------------------------------------------- #
# Levenberg-Marquardt with numerical Jacobian (port of lmsolve / numjac)
# --------------------------------------------------------------------------- #
def _numjac(resfun, x, r0):
    n = x.size
    m = r0.size
    J = np.zeros((m, n))
    for k in range(n):
        dk = 1e-6 * max(abs(x[k]), 1e-6)
        xp = x.copy()
        xp[k] += dk
        J[:, k] = (resfun(xp) - r0) / dk
    return J


def _lmsolve(resfun, x0, maxiter, tol):
    x = np.asarray(x0, dtype=float).copy()
    r = resfun(x)
    cost = r @ r
    mu = 1e-3
    for _ in range(maxiter):
        J = _numjac(resfun, x, r)
        H = J.T @ J
        g = J.T @ r
        while True:
            damp = mu * np.diag(np.maximum(np.diag(H), 1e-12))
            dx = -np.linalg.solve(H + damp, g)
            xn = x + dx
            rn = resfun(xn)
            cn = rn @ rn
            if cn < cost:
                x, r = xn, rn
                mu = max(mu / 3.0, 1e-12)
                if np.linalg.norm(dx) <= tol * (np.linalg.norm(x) + tol):
                    return x
                cost = cn
                break
            mu *= 3.0
            if mu > 1e12:
                return x
    return x


def _cmif_peaks(G, f, nflex):
    N = f.size
    sv = np.array([np.linalg.svd(G[:, :, k], compute_uv=False)[0] for k in range(N)])
    lsv = 20 * np.log10(sv)
    inner = (lsv[1:-1] > lsv[:-2]) & (lsv[1:-1] > lsv[2:])
    pk = np.where(inner)[0] + 1
    order = np.argsort(-lsv[pk])
    pk = pk[order]
    if pk.size < nflex:
        fpk = np.logspace(np.log10(f[1]), np.log10(f[-1]), nflex + 2)[1:-1]
    else:
        fpk = np.sort(f[pk[:nflex]])
    return fpk


def _modal2ss(modal, nrbm, nflex, gen):
    if gen:
        ny = modal["psil"].shape[0]
        nu = modal["psir"].shape[0]
    else:
        ny = modal["phil"].shape[0]
        nu = modal["phir"].shape[0]
    blocks = []
    Brows = []
    Ccols = []
    for j in range(nrbm):
        blocks.append(np.array([[0.0, 1.0], [0.0, 0.0]]))
        Brows.append(np.vstack([np.zeros((1, nu)), modal["phir_rbm"][:, j][None, :]]))
        Ccols.append(np.column_stack([modal["phil_rbm"][:, j], np.zeros(ny)]))
    for i in range(nflex):
        w = 2 * np.pi * modal["wn"][i]
        z = modal["zeta"][i]
        if gen:
            lam = modal["lambda"][i]
            blocks.append(np.array([[lam.real, -lam.imag], [lam.imag, lam.real]]))
            Brows.append(np.vstack([np.real(modal["psir"][:, i])[None, :],
                                    np.imag(modal["psir"][:, i])[None, :]]))
            Ccols.append(np.column_stack([2 * np.real(modal["psil"][:, i]),
                                          -2 * np.imag(modal["psil"][:, i])]))
        else:
            blocks.append(np.array([[0.0, 1.0], [-w ** 2, -2 * z * w]]))
            Brows.append(np.vstack([np.zeros((1, nu)),
                                    np.real(modal["phir"][:, i])[None, :]]))
            Ccols.append(np.column_stack([np.real(modal["phil"][:, i]), np.zeros(ny)]))
    A = sla.block_diag(*blocks) if blocks else np.zeros((0, 0))
    B = np.vstack(Brows) if Brows else np.zeros((0, nu))
    C = np.hstack(Ccols) if Ccols else np.zeros((ny, 0))
    return control.ss(A, B, C, np.real(modal["D"]))


def frf2modal(Pest, nrbm, nflex, damping="proportional", initfreq=None,
              initdamp=0.01, feedthrough=True, weight="invmag", band=None,
              maxiter=100, tol=1e-8):
    """Identify a modal model from a measured (MIMO) FRF.  See module docstring."""
    gen = damping.lower() == "general"

    if isinstance(Pest, FrfData):
        G = Pest.response
        f = np.asarray(Pest.freq, dtype=float)
    else:                                   # (response, freq) pair
        G, f = Pest
        G = np.asarray(G)
        f = np.asarray(f, dtype=float)
    if band is not None:
        m = (f >= band[0]) & (f <= band[1])
        G, f = G[:, :, m], f[m]
    ny, nu, N = G.shape
    w = 2 * np.pi * f
    xi = 1j * w
    if weight.lower() == "invmag":
        Wt = 1.0 / np.maximum(np.abs(G), np.finfo(float).eps)
    else:
        Wt = np.ones((ny, nu, N))
    hasD = bool(feedthrough)

    # ---- initial poles --------------------------------------------------
    if initfreq is None:
        wn0 = 2 * np.pi * _cmif_peaks(G, f, nflex)
    else:
        wn0 = 2 * np.pi * np.atleast_1d(np.asarray(initfreq, dtype=float))
    ze0 = np.atleast_1d(np.asarray(initdamp, dtype=float))
    if ze0.size == 1:
        ze0 = ze0 * np.ones(nflex)

    # ---- STAGE 1 : additive model --------------------------------------
    p0 = np.concatenate([wn0, ze0])
    r1 = lambda p: _stage1_residual(p, G, xi, Wt, nrbm, nflex, hasD, gen)
    p1 = _lmsolve(r1, p0, maxiter, tol)
    _, C = _stage1_residual(p1, G, xi, Wt, nrbm, nflex, hasD, gen, want_C=True)
    wn = p1[:nflex]
    ze = p1[nflex:]
    Dest, Rrbm_full, B0, B1 = _unpack_residues(C, ny, nu, nrbm, nflex, hasD, gen)

    # ---- rank-one SVD initialisation -----------------------------------
    phil_r = np.zeros((ny, nrbm))
    phir_r = np.zeros((nu, nrbm))
    if nrbm > 0:
        U, S, Vh = np.linalg.svd(np.real(Rrbm_full))
        for j in range(nrbm):
            phil_r[:, j] = U[:, j] * np.sqrt(S[j])
            phir_r[:, j] = Vh[j, :] * np.sqrt(S[j])

    if gen:
        psil = np.zeros((ny, nflex), dtype=complex)
        psir = np.zeros((nu, nflex), dtype=complex)
        for i in range(nflex):
            lam = -ze[i] * wn[i] + 1j * wn[i] * np.sqrt(max(1 - ze[i] ** 2, 0.0))
            Li = (B0[:, :, i] + lam * B1[:, :, i]) / (lam - np.conj(lam))
            U, S, Vh = np.linalg.svd(Li)
            psil[:, i] = U[:, 0] * np.sqrt(S[0])
            psir[:, i] = Vh[0, :] * np.sqrt(S[0])      # conj(V[:,0]) == Vh[0,:]
    else:
        phil = np.zeros((ny, nflex))
        phir = np.zeros((nu, nflex))
        for i in range(nflex):
            U, S, Vh = np.linalg.svd(np.real(B0[:, :, i]))
            phil[:, i] = U[:, 0] * np.sqrt(S[0])
            phir[:, i] = Vh[0, :] * np.sqrt(S[0])

    # ---- STAGE 2 : structured refinement -------------------------------
    if gen:
        rho0 = np.concatenate([wn, ze, _F(np.real(psil)), _F(np.imag(psil)),
                               _F(np.real(psir)), _F(np.imag(psir)),
                               _F(phil_r), _F(phir_r)])
    else:
        rho0 = np.concatenate([wn, ze, _F(phil), _F(phir),
                               _F(phil_r), _F(phir_r)])
    if hasD:
        rho0 = np.concatenate([rho0, _F(np.real(Dest))])
    r2 = lambda r: _stage2_residual(r, G, xi, Wt, ny, nu, nrbm, nflex, hasD, gen)
    rho = _lmsolve(r2, rho0, maxiter, tol)

    # ---- unpack & normalize --------------------------------------------
    wn, ze, ms, phil_r, phir_r, Dest = _unpack_modal(rho, ny, nu, nrbm, nflex, hasD, gen)
    modal = {"damping": damping, "wn": wn / (2 * np.pi), "zeta": ze,
             "phil_rbm": phil_r, "phir_rbm": phir_r, "D": Dest}
    modal["Rrbm"] = np.zeros((ny, nu, nrbm))
    for j in range(nrbm):
        modal["Rrbm"][:, :, j] = np.outer(phil_r[:, j], phir_r[:, j])

    if gen:
        psil, psir = ms["psil"], ms["psir"]
        for i in range(nflex):
            a = np.linalg.norm(psir[:, i])
            if a > 0:
                psir[:, i] /= a
                psil[:, i] *= a
            im = np.argmax(np.abs(psir[:, i]))
            ph = np.angle(psir[im, i])
            psir[:, i] *= np.exp(-1j * ph)
            psil[:, i] *= np.exp(1j * ph)
        modal["psil"], modal["psir"] = psil, psir
        wn_hz = modal["wn"]
        modal["lambda"] = (-ze * (2 * np.pi * wn_hz)
                           + 1j * (2 * np.pi * wn_hz) * np.sqrt(np.maximum(1 - ze ** 2, 0)))
    else:
        phil, phir = ms["phil"], ms["phir"]
        for i in range(nflex):
            a = np.linalg.norm(phir[:, i])
            if a > 0:
                phir[:, i] /= a
                phil[:, i] *= a
            if phil[0, i] < 0:
                phil[:, i] = -phil[:, i]
                phir[:, i] = -phir[:, i]
        modal["phil"], modal["phir"] = phil, phir

    Pm = _modal2ss(modal, nrbm, nflex, gen)
    return modal, Pm
