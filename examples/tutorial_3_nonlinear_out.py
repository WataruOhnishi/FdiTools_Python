"""Tutorial 3 (output non-linearity) - Python port of
``Examples/Tutorial_3_nonlinear_out.m`` (which used the Simulink model
``model_nl_out.slx``).

That Simulink model feeds a polynomial of the **output** back to the plant
input:

    u_plant = u + even_nl * y^2 + odd_nl * y^3
    y       = G * u_plant                       (out.out = y)

Since G is strictly proper (no direct feed-through) there is no algebraic loop,
so the block diagram is equivalent to the state-space ODE

    dx/dt = A x + B ( u(t) + even_nl*(Cx)^2 + odd_nl*(Cx)^3 ),   y = C x

which we integrate here with fixed-step RK4 — the exact equivalent of the
Simulink simulation.
"""

import numpy as np
import control
import matplotlib.pyplot as plt

import fditools as fdi
from _data import benchmark_plant
from _plot import save_fig, show


def simulate_nl_out(P0, u, fs, odd_nl, even_nl):
    """Integrate the output-feedback non-linear plant (RK4, dt = 1/fs)."""
    sys = control.ss(P0)
    A = np.asarray(sys.A, dtype=float)
    B = np.asarray(sys.B, dtype=float).reshape(-1)
    C = np.asarray(sys.C, dtype=float).reshape(-1)
    n = A.shape[0]
    dt = 1.0 / fs

    def deriv(x, uin):
        yv = C @ x                       # strictly proper -> y = C x
        utot = uin + even_nl * yv ** 2 + odd_nl * yv ** 3
        return A @ x + B * utot

    x = np.zeros(n)
    y = np.empty(u.size)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(u.size):
            y[k] = C @ x
            if not np.isfinite(y[k]):          # feedback nl diverged
                return y[:k], False
            uk = u[k]
            uk1 = u[k + 1] if k + 1 < u.size else u[k]
            uh = 0.5 * (uk + uk1)
            k1 = deriv(x, uk)
            k2 = deriv(x + 0.5 * dt * k1, uh)
            k3 = deriv(x + 0.5 * dt * k2, uh)
            k4 = deriv(x + dt * k3, uk1)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y, True


def main():
    P0, label = benchmark_plant()
    print(f"true plant: {label}")

    harm = dict(fs=10000.0, df=1.0, fl=1.0, fh=1000.0, fr=1.02)
    options = dict(itp="r", ctp="c", dtp="O", gtp="l")     # odd-odd, linear grid
    ms = fdi.multisine(harm, control.tf([1], [1]), options)
    fs = harm["fs"]
    base = np.squeeze(ms.x[0, 0, :])

    nrofp = 20
    u0 = np.tile(base, nrofp)
    odd_nl, even_nl = 0.1, 1.0
    out_noise = 1e-4 * np.random.default_rng(0).standard_normal(u0.size)

    cases = [("amp 1", 1.0), ("amp 0.1", 0.1), ("amp 10", 10.0), ("noise-free", 1.0)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, amp) in zip(axs.ravel(), cases):
        u = amp * u0
        if name == "noise-free":
            y = control.forced_response(P0, np.arange(u.size) / fs, u).outputs
        else:
            y, ok = simulate_nl_out(P0, u, fs, odd_nl, even_nl)
            if not ok:
                ax.text(0.5, 0.5, f"{name}: feedback diverged\n(stable with the real "
                                  "benchmark plant)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(name)
                continue
            y = y + out_noise

        x, _ = fdi.pretreat(u, ms.nrofs, fs, 2, 0)
        yy, _ = fdi.pretreat(y, ms.nrofs, fs, 2, 0)
        Yl, fl_, Yo, fo, Ye, fe, Yn, fn = fdi.time2nld(x, yy, fs, harm["fl"], harm["fh"], harm["df"])
        ax.semilogx(fl_, fdi.dbm(Yl[:, 0]), "*", ms=3, label="Y lin")
        ax.semilogx(fe, fdi.dbm(Ye[:, 0]), "*", ms=3, label="Y even")
        ax.semilogx(fo, fdi.dbm(Yo[:, 0]), "*", ms=3, label="Y odd")
        ax.semilogx(fn, fdi.dbm(Yn[:, 0]), "*", ms=3, label="Y noise")
        ax.set_xlim(harm["fl"], harm["fh"])
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title(name)
    for ax in axs.ravel():                     # legend on the first panel with data
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2)
            break
    fig.suptitle(f"Output non-linearity, Simulink-equivalent ({label})")
    save_fig(fig, "tutorial_3_nonlinear_out.png")
    show()


if __name__ == "__main__":
    main()
