"""erasers/backends public API."""

from .aot_gan import AOTGANBackend
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
    "AOTGANBackend",
    "RemoteInpaintBackend",
    "CfSd15InpaintBackend",
    "Flux2KleinInpaintBackend",
]
