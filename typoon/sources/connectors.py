"""Available manga source connectors."""

from __future__ import annotations


def get_connectors():
    from .comix import ComixConnector

    return [ComixConnector()]
