"""Global test configuration.

Ensures the test suite never opens an interactive plot window. Several test
helpers (e.g. ``try_agents``) default to ``draw=True`` and call
``world.draw()`` followed by ``matplotlib.pyplot.show()`` (and some negmas code
paths render with plotly). On a machine with a GUI/browser available those
calls block the run until the user manually dismisses the figure. Forcing a
non-interactive matplotlib backend and turning ``show`` into a no-op keeps the
suite headless and non-blocking everywhere.
"""

from __future__ import annotations

import os

# Force a non-interactive backend before pyplot is imported anywhere. This must
# happen at import time of conftest (which pytest loads before any test module).
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Even with the Agg backend ``show()`` is a harmless no-op, but make the
    # intent explicit and guard against any backend that might still block.
    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
except Exception:  # pragma: no cover - matplotlib should always be present
    pass

try:
    # Prevent plotly figures from opening a browser tab during tests.
    import plotly.io as pio

    pio.renderers.default = "json"

    import plotly.graph_objects as go

    go.Figure.show = lambda *args, **kwargs: None  # type: ignore[assignment]
except Exception:  # pragma: no cover - plotly may not be installed
    pass
