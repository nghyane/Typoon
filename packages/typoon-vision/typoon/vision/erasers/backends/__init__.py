"""erasers/backends public API."""

from ..contracts import InpaintBackend
from .remote import (
    CfSd15InpaintBackend,
    Flux2KleinInpaintBackend,
    RemoteInpaintBackend,
)
from .telea import TeLeABackend

__all__ = [
    "InpaintBackend",
    "TeLeABackend",
    "RemoteInpaintBackend",
    "CfSd15InpaintBackend",
    "Flux2KleinInpaintBackend",
]
