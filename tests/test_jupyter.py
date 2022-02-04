import os
from pathlib import Path

import papermill as pm
import pytest

SCML_IGNORE_TEST_NOTEBOOKS = os.environ.get("SCML_TEST_NOTEBOOKS", False)


def notebooks():
    base = Path(__file__).parent.parent / "notebooks"
    return list(_ for _ in base.glob("**/*.ipynb") if "checkpoints" not in str(_))


@pytest.mark.skipif(
    condition=SCML_IGNORE_TEST_NOTEBOOKS,
    reason="No Genius Bridge, skipping genius-agent tests",
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
