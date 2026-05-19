"""RemoteInpaintBackend — HTTP POST {image, mask} → inpainted image.

Skeleton for any cloud inpainting service:
  - Cloudflare Workers AI SD1.5 inpainting (available now)
  - BFL API FLUX.2 inpainting (when available)
  - Self-hosted service

Contract: POST multipart/form-data or JSON to `url` with
  image  — PNG bytes or base64
  mask   — PNG bytes or base64 (255=inpaint region)
  prompt — optional hint for diffusion models

The subclass only needs to implement `_build_request` and
`_parse_response` to adapt to each service's API shape.
"""

from __future__ import annotations

import base64
import logging
from abc import abstractmethod
from io import BytesIO
from typing import Any

import cv2
import httpx
import numpy as np

from ..contracts import InpaintBackend

__all__ = [
    "RemoteInpaintBackend",
    "CfSd15InpaintBackend",
    "Flux2KleinInpaintBackend",
]

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30.0  # seconds


def _encode_png_b64(image_rgb: np.ndarray) -> str:
    bgr    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _decode_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_png_bytes(image_rgb: np.ndarray) -> bytes:
    bgr    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


class RemoteInpaintBackend:
    """Base class for HTTP-based inpaint services.

    Subclasses implement:
      _build_request(image_rgb, mask, **kwargs) → (method, url, kwargs)
        where kwargs are passed directly to httpx.request().
      _parse_response(response) → np.ndarray (RGB)
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
        self._url        = url
        self._token      = auth_token
        self._timeout    = timeout
        self._extra      = extra or {}

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


# ─── Cloudflare Workers AI SD 1.5 Inpainting ─────────────────────────────────


class CfSd15InpaintBackend(RemoteInpaintBackend):
    """Cloudflare Workers AI Stable Diffusion 1.5 Inpainting.

    Model: @cf/runwayml/stable-diffusion-v1-5-inpainting
    API:   POST multipart/form-data
           image   — PNG bytes  (source image)
           mask    — PNG bytes  (255 = region to inpaint)
           prompt  — string     (description of what to fill)
    Response: {"image": "<base64 PNG>"}

    Usage:
        backend = CfSd15InpaintBackend(
            account_id="...",
            auth_token="cf-...",
            prompt="manga panel background, clean screentone",
        )
    """

    name = "cf_sd15_inpaint"

    _MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"

    def __init__(
        self,
        account_id: str,
        auth_token: str,
        *,
        prompt: str = "manga panel background, clean",
        num_steps: int = 20,
        timeout: float = 30.0,
    ) -> None:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/{self._MODEL}"
        )
        super().__init__(url, auth_token=auth_token, timeout=timeout)
        self._prompt    = prompt
        self._num_steps = num_steps

    def _build_request(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, str, dict]:
        # mask must be grayscale PNG — already uint8 (H,W) 255/0
        mask_png   = _encode_png_bytes(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        )
        image_png  = _encode_png_bytes(image_rgb)
        files = {
            "image":     ("image.png",  image_png, "image/png"),
            "mask":      ("mask.png",   mask_png,  "image/png"),
            "prompt":    (None, self._prompt),
            "num_steps": (None, str(self._num_steps)),
        }
        return "POST", self._url, {"files": files}

    def _parse_response(self, response: httpx.Response) -> np.ndarray:
        data = response.json()
        # CF returns {"result": {"image": "<b64>"}, "success": true}
        b64 = (data.get("result") or data).get("image", "")
        return _decode_b64_png(b64)


# ─── FLUX.2 Klein Inpainting (future) ────────────────────────────────────────


class Flux2KleinInpaintBackend(RemoteInpaintBackend):
    """FLUX.2 Klein inpainting via BFL API or self-hosted endpoint.

    Currently a skeleton — BFL/CF inpainting API not yet public.
    Replace _build_request and _parse_response when endpoint is available.

    Expected API (speculative, based on BFL edit endpoint patterns):
      POST multipart/form-data
        image_url or image (binary)
        mask_url  or mask  (binary, 255=inpaint)
        prompt    (optional — FLUX edit is prompt-guided)
        steps     (default 4 for klein)
    """

    name = "flux2_klein_inpaint"

    def __init__(
        self,
        url: str,
        auth_token: str,
        *,
        prompt: str = "",
        steps: int = 4,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(url, auth_token=auth_token, timeout=timeout)
        self._prompt = prompt
        self._steps  = steps

    def _build_request(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[str, str, dict]:
        # TODO: update when BFL/CF exposes inpainting endpoint
        raise NotImplementedError(
            "FLUX.2 Klein inpainting endpoint not yet available. "
            "Implement _build_request when the API is public."
        )

    def _parse_response(self, response: httpx.Response) -> np.ndarray:
        raise NotImplementedError
