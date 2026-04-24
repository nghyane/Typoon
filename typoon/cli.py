"""CLI entry point — redirects to interfaces/cli.py.

Kept for backward compatibility with pyproject.toml [project.scripts].
"""

from .interfaces.cli import app  # noqa: F401
