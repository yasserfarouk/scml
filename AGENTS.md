# AGENTS.md - AI Coding Agent Instructions for SCML

This document provides essential information for AI coding agents working on the SCML (ANAC Supply Chain Management League) codebase.

## Project Overview

- **Language:** Python 3.11+ (supports 3.12)
- **Description:** Official platform for running ANAC Supply Chain Management Leagues
- **Repository:** https://github.com/yasserfarouk/scml
- **Documentation:** https://scml.readthedocs.io/

## Project Structure

```
src/scml/                    # Main source code
├── __init__.py              # Package init, exports from submodules
├── cli.py                   # Main CLI (entry point: scml)
├── cliadv.py                # Advanced CLI (entry point: cliadv)
├── oneshot/                 # OneShot game mode (single-step negotiations)
│   ├── agent.py             # Base agent classes
│   ├── agents/              # Agent implementations
│   ├── awi.py               # Agent World Interface
│   ├── world.py             # World simulation
│   ├── ufun.py              # Utility functions
│   └── rl/                  # Reinforcement learning components
├── std/                     # Standard game mode (multi-step)
│   ├── agent.py, agents/, awi.py, world.py, rl/
├── scml2019/                # 2019 competition code (legacy)
├── scml2020/                # 2020 competition code
└── vendor/                  # Vendored dependencies
tests/                       # Test suite (pytest + hypothesis)
├── oneshot/, std/, rl/, core/, etc/
└── switches.py              # Test configuration flags
```

## Build/Lint/Test Commands

### Installation (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Or install specific dependency groups
uv pip install -e .              # Runtime dependencies only
uv pip install -e ".[dev]"       # With dev dependencies
uv pip install -e ".[docs]"      # With docs dependencies
uv pip install -e ".[dev,docs]"  # All dependencies
```

### Running Tests

```bash
# Run all tests
uv run pytest tests

# Run a single test file
uv run pytest tests/oneshot/test_scml2024oneshot.py -v

# Run a single test function
uv run pytest tests/oneshot/test_scml2024oneshot.py::test_run_single_agent -v

# Run tests matching a keyword
uv run pytest -k test_myfeature

# Run with coverage
uv run pytest --cov=src/scml --cov-report=term-missing
```

### Test Environment Variables (tests/switches.py)

- `SCML_RUNALL_TESTS` - Run all tests including slow ones
- `SCML_FASTRUN` - Fast test mode (skip slow tests)
- `SCML_RUN2021_ONESHOT` - Run 2021 OneShot tests
- `SCML_RUN2021_STD` - Run 2021 Standard tests

### Linting and Formatting

```bash
# Run all pre-commit hooks (ruff, formatting, etc.)
uv run pre-commit run --all-files

# Run ruff directly
uv run ruff check --fix src/
uv run ruff format src/
```

## Code Style Guidelines

### Formatting

- **Formatter:** Ruff (configured in pyproject.toml)
- **Line length:** 140 characters
- **Indentation:** 4 spaces
- **Line endings:** LF (Unix-style)
- **Encoding:** UTF-8
- **Final newline:** Required
- **Trailing whitespace:** Not allowed

### Imports

```python
from __future__ import annotations  # Use at top of files for modern type hints

# Standard library imports
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

# Third-party imports (negmas is the core dependency)
from negmas import Contract, Entity, Issue, Outcome

# Local imports with TYPE_CHECKING guard for circular imports
if TYPE_CHECKING:
    from scml.oneshot import OneShotAWI, OneShotUFun

# Define __all__ for public API
__all__ = ["OneShotAgent", "OneShotSyncAgent"]
```

### Type Hints

- Use modern Python 3.10+ type hint syntax: `list[str]` not `List[str]`
- Use `X | None` instead of `Optional[X]`
- Use `from __future__ import annotations` for forward references
- Guard type-only imports with `if TYPE_CHECKING:`
- Pyright is configured for type checking (pyproject.toml)

### Naming Conventions

- **Classes:** PascalCase (`OneShotAgent`, `SCML2024OneShotWorld`)
- **Functions/methods:** snake_case (`make_ufun`, `before_step`)
- **Constants:** SCREAMING_SNAKE_CASE (`PLACEHOLDER_AGENT_PREFIX`)
- **Private attributes:** Single underscore prefix (`self._awi`, `self._owner`)
- **Module-level private:** Single underscore (`_helper_function`)

### Docstrings

Use Google-style docstrings with Remarks section for implementation notes:

```python
def make_ufun(self, add_exogenous=False):
    """
    Creates a utility function for the agent.

    Args:
        add_exogenous: If `True` then the exogenous contracts of the agent
                       will be automatically added whenever the ufun is
                       evaluated for any set of contracts, offers or otherwise.

    Remarks:
        - You can always assume that self.ufun returns the ufun for your agent.
        - You will not need to directly use this method in most cases.
    """
```

### Error Handling

- Use assertions for internal invariants: `assert self._owner is not None`
- Raise `ValueError` for invalid arguments with descriptive messages
- Use warnings module for non-fatal issues: `warnings.warn("message")`

```python
@property
def awi(self) -> OneShotAWI:
    if not self._awi:
        raise ValueError("The AWI is not assigned yet.")
    return self._awi
```

### Class Structure

1. Class docstring
2. `__init__` method
3. Properties (`@property`)
4. Public methods
5. Protected/private methods
6. Abstract methods at the end with `@abstractmethod`

### Testing Patterns

- Use pytest with parametrize for multiple test cases
- Use hypothesis for property-based testing
- Test files: `test_*.py` or `*_test.py`
- Import from parent test modules: `from ..switches import DefaultOneShotWorld`

```python
@pytest.mark.parametrize(
    "atype,no_bankrupt,some_profits",
    [
        (SyncRandomOneShotAgent, True, True),
        (RandomOneShotAgent, False, False),
    ],
)
def test_run_single_agent(atype, no_bankrupt, some_profits):
    world = SCML2024OneShotWorld(**SCML2024OneShotWorld.generate(atype, n_steps=50))
    world.run()
    # assertions...
```

## Key Dependencies

- **negmas** (>=0.15.2): Core negotiation framework - most classes inherit from negmas
- **click/click-aliases**: CLI framework
- **pytest/hypothesis**: Testing
- **joblib**: Parallel execution
- **gymnasium**: RL environments

## CLI Entry Points

- `scml` - Main CLI command (scml.cli:main)
- `cliadv` - Advanced CLI (scml.cliadv:cli)

## Pre-commit Hooks

The project uses these pre-commit hooks:
- debug-statements, check-yaml, end-of-file-fixer, trailing-whitespace
- ruff (linting + formatting with --fix --unsafe-fixes)
- blacken-docs (format code in documentation)
- rst-linter (validate RST files)

Always run `uv run pre-commit run --all-files` before committing.
