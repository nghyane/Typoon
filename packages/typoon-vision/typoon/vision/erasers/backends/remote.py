"""HTTP-based inpaint backends.

RemoteInpaintBackend
  Abstract base for any inpaint service reached over HTTP. Subclasses
  implement `_build_request` and `_parse_response` to adapt to each
  service's wire format.

TyphoonInpaintBackend
  Concrete client for the Typoon Rust/Candle inpaint container at
  spike/inpaint. Raw bytes protocol:
    POST /inpaint?w=<W>&h=<H>
    body = RGB(W·H·3) ++ mask(W·H)
    response = RGB(W·H·3)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import httpx
import numpy as np

from ..contracts import InpaintBackend


__all__ = ["RemoteInpaintBackend", "TyphoonInpaintBackend"]

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30.0  # seconds


class RemoteInpaintBackend:
    """Base class for HTTP-based inpaint services.

    Subclasses implement:
      _build_request(image_rgb, mask) → (method, url, httpx_kwargs)
      _parse_response(response)       → np.ndarray (RGB)
    """

    name = "remote"

    def __init__(
        self,
        url: str,
        *,
        auth_token: str | None = None,
        timeout: float = _REQUEST_TIMEOUT,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._url     = url
        self._token   = auth_token
        self._timeout = timeout
        self._extra   = extra or {}

    @abstractmethod
    def _build_request(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, str, dict]:
        """Return (method, url, httpx_kwargs)."""
        ...

    @abstractmethod
    def _parse_response(self, response: httpx.Response) -> np.ndarray:
        """Parse HTTP response → RGB image."""
        ...

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        method, url, kwargs = self._build_request(image_rgb, mask)
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.request(method, url, headers=headers, **kwargs)
                resp.raise_for_status()
            return self._parse_response(resp)
        except Exception as exc:
            logger.warning("RemoteInpaintBackend %s failed: %s", self.name, exc)
            raise


class TyphoonInpaintBackend(RemoteInpaintBackend):
    """Typoon Rust/Candle inpaint container (spike/inpaint).

    Wire format (matches crates/inpaint/bin/serve.rs):
      POST {url}/inpaint?w=<W>&h=<H>
      body  = RGB bytes (W·H·3) ++ mask bytes (W·H)
      reply = RGB bytes (W·H·3) inpainted

    Both width and height must be multiples of 8 (model's pad_mod).
    The caller (TiledInpainter) handles padding; this backend assumes
    valid dimensions.
    """

    name = "typoon_inpaint"

    def _build_request(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, str, dict]:
        h, w = image_rgb.shape[:2]
        body = image_rgb.tobytes() + mask.tobytes()
        url  = f"{self._url.rstrip('/')}/inpaint?w={w}&h={h}"
        return "POST", url, {
            "content": body,
            "headers": {"Content-Type": "application/octet-stream"},
        }

    def _parse_response(self, response: httpx.Response) -> np.ndarray:
        # The Rust server returns raw RGB bytes; reshape via the dimensions
        # we requested. httpx doesn't carry that back, so the caller resizes
        # if needed — here we assume the caller passes square tiles whose
        # shape it can reconstruct. Simpler: include shape in caller.
        raise NotImplementedError(
            "TyphoonInpaintBackend.inpaint overrides _parse_response path"
        )

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Override inpaint() directly so we keep H, W in scope for reshape.
        h, w = image_rgb.shape[:2]
        method, url, kwargs = self._build_request(image_rgb, mask)
        headers = kwargs.pop("headers", {})
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.request(method, url, headers=headers, **kwargs)
                resp.raise_for_status()
            body = resp.read()
        except Exception as exc:
            logger.warning("TyphoonInpaintBackend failed: %s", exc)
            raise
        expected = h * w * 3
        if len(body) != expected:
            raise ValueError(
                f"Typoon inpaint: body {len(body)} bytes != expected {expected}"
            )
        return np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)
