"""Tiny plotting helper for the example scripts: save a PNG next to the
examples and (optionally) show interactive windows."""

from __future__ import annotations

import os


def save_fig(fig, name):
    """Save *fig* as ``examples/<name>`` and print the path."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"figure saved to: {out}")
    return out


def show():
    """Pop up the figure windows and block until they are closed.

    Set the environment variable ``FDI_NOSHOW=1`` to skip the (blocking) window
    display — figures are still saved as PNG, so several scripts can be run
    back-to-back without closing windows in between.
    """
    if os.environ.get("FDI_NOSHOW"):
        return
    import matplotlib.pyplot as plt
    plt.show()
