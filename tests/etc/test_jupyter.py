from pathlib import Path

import papermill as pm
import pytest

from ..switches import SCML_RUN_NOTEBOOKS


def notebooks():
    base = Path(__file__).parent.parent / "notebooks"
    return list(_ for _ in base.glob("**/*.ipynb") if "checkpoints" not in str(_))


@pytest.mark.skipif(
    condition=not SCML_RUN_NOTEBOOKS,
    reason="Environment set to ignore running notebook tests. See switches.py",
)
@pytest.mark.parametrize("notebook", notebooks())
def test_notebook(notebook):
    base = Path(__file__).parent.parent / "notebooks"
    dst = notebook.relative_to(base)
    dst = Path(__file__).parent / "tmp_notebooks" / str(dst)
    dst.parent.mkdir(exist_ok=True, parents=True)
    pm.execute_notebook(
        notebook,
        dst,
    )
