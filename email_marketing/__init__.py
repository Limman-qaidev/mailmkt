"""Top‑level package for the mailmkt application.

This package provides a modular implementation of an email marketing platform.
Individual subpackages handle specific concerns such as sending mail, tracking
engagement events, presenting a dashboard, and orchestrating background
workflows.  The entire codebase is written for Python 3.13 with type
annotations and aims to pass ``mypy`` in strict mode.

The ``__all__`` variable enumerates the primary public modules for
convenience when using ``from email_marketing import ...``.
"""

from __future__ import annotations

__all__ = [
    "app",
    "mailer",
    "tracking",
    "dashboard",
    "workflows",
    "ab_testing",
]

# SemVer version of the package
__version__: str = "0.1.0"
