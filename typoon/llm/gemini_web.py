"""Google Gemini *web* provider — driven by HanaokaYuzu's reverse of
`gemini.google.com`. Talks straight to the consumer chat backend over HTTPS
using cookies obtained from a logged-in browser. No API key needed.

Why this provider exists
------------------------
The official Gemini API tier on Google One AI Pro has aggressive per-minute
quotas (especially for image inputs). The web frontend shares the same model
family but uses an account-level daily budget instead, which is more forgiving
for batch translation. This adapter lets a Typoon deployment route either the
translation or vision_agent stage through that budget while leaving every other
provider untouched.

Capabilities and limits (vs. native `gemini` provider)
------------------------------------------------------
- Text in, text out: yes.
- Image input: yes (multi-image per turn).
- Tool / function calling: NO. The web frontend does not expose the function
  calling surface and the reverse library does not emulate it. If a stage
  passes a non-empty ``tools`` list we raise `OperatorActionRequired` so the
  operator notices in the pause queue instead of silently dropping calls.
- System prompt: web Gemini has no role separation, so we concatenate the
  SYSTEM message into the head of the first USER message.
- Conversation memory: each ``call()`` is one-shot — the adapter starts a
  fresh `ChatSession` per call so stages remain stateless.

Cookie sourcing
---------------
- Explicit values via config (`secure_1psid`, `secure_1psidts`).
- Or auto-load from a local browser profile (`cookie_browser = "edge"|"chrome"|...`).
  Uses the library's `load_browser_cookies` helper, which depends on the
  optional `browser-cookie3` package being installed.

The library auto-rotates ``__Secure-1PSIDTS`` every 10 minutes (see
`auto_refresh=True` by default).
"""

from __future__ import annotations

import asyncio
import base64
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ._retry import with_retry
from .errors import OperatorActionRequired, UpstreamUnavailable
from .ir import (
    CallResponse,
    ContentPart,
    Message,
    Role,
    ToolDef,
)

# Import lazily inside the class so that a Typoon deployment that doesn't use
# this provider doesn't pay the dependency cost.


_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.S)


def _is_retryable(exc: BaseException) -> bool:
    # The reverse library wraps a `curl_cffi` session. Transport-level failures
    # (connect/read/timeout) and 5xx from Google can clear on retry; outright
    # auth errors cannot.
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException)):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "temporar" in name:
        return True
    msg = str(exc).lower()
    if "503" in msg or "502" in msg or "504" in msg or "rate" in msg:
        return True
    return False


def _is_auth_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "authentication" in msg
        or "1psid" in msg
        or "cookies" in msg
        or "unauthorized" in msg
        or "auth" in type(exc).__name__.lower()
    )


class GeminiWebProvider:
    """Provider adapter for gemini.google.com via HanaokaYuzu/Gemini-API."""

    __slots__ = (
        "_model_name",
        "_secure_1psid",
        "_secure_1psidts",
        "_cookie_browser",
        "_cookie_file",
        "_client",
        "_init_lock",
    )

    def __init__(
        self,
        *,
        model: str,
        secure_1psid: str | None = None,
        secure_1psidts: str | None = None,
        cookie_browser: str | None = None,
        cookie_file: str | None = None,
    ) -> None:
        self._model_name = model
        self._secure_1psid = secure_1psid or None
        self._secure_1psidts = secure_1psidts or None
        self._cookie_browser = cookie_browser or None
        self._cookie_file = cookie_file or None
        self._client: Any = None  # GeminiClient, lazy
        self._init_lock = asyncio.Lock()

    # ---- lifecycle ---------------------------------------------------------

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._init_lock:
            if self._client is not None:
                return self._client
            try:
                from gemini_webapi import GeminiClient
            except ImportError as exc:
                raise OperatorActionRequired(
                    "gemini_web provider requires the `gemini-webapi` package; "
                    "install it via `pip install gemini-webapi browser-cookie3`."
                ) from exc

            secure_1psid, secure_1psidts = self._load_cookies()
            client = GeminiClient(
                secure_1psid=secure_1psid,
                secure_1psidts=secure_1psidts,
            )
            try:
                await client.init(timeout=300, auto_close=False, auto_refresh=True)
            except Exception as exc:
                if _is_auth_failure(exc):
                    raise OperatorActionRequired(
                        "gemini_web: failed to authenticate with provided cookies. "
                        "Re-export __Secure-1PSID and __Secure-1PSIDTS from a "
                        "logged-in browser, or set `cookie_browser` to auto-load."
                    ) from exc
                raise UpstreamUnavailable(f"gemini_web init failed: {exc}") from exc
            self._client = client
            return client

    def _load_cookies(self) -> tuple[str | None, str | None]:
        """Return (1psid, 1psidts), preferring explicit config over auto-load."""
        if self._secure_1psid:
            return self._secure_1psid, self._secure_1psidts
        if self._cookie_browser or self._cookie_file:
            try:
                import browser_cookie3 as bc3
            except ImportError as exc:
                raise OperatorActionRequired(
                    "gemini_web: cookie_browser/cookie_file requires `browser-cookie3`."
                ) from exc

            browser = (self._cookie_browser or "edge").lower()
            fn = getattr(bc3, browser, None)
            if fn is None:
                raise OperatorActionRequired(
                    f"gemini_web: unknown cookie_browser={browser!r}. "
                    f"Supported: chrome, chromium, edge, brave, opera, opera_gx, vivaldi, firefox, librewolf, safari."
                )
            kwargs: dict[str, Any] = {"domain_name": ".google.com"}
            if self._cookie_file:
                kwargs["cookie_file"] = self._cookie_file
            try:
                jar = fn(**kwargs)
            except Exception as exc:
                raise OperatorActionRequired(
                    f"gemini_web: failed to read cookies via browser_cookie3.{browser}: {exc}"
                ) from exc

            psid = next((c.value for c in jar if c.name == "__Secure-1PSID"), None)
            psidts = next((c.value for c in jar if c.name == "__Secure-1PSIDTS"), None)
            if not psid:
                raise OperatorActionRequired(
                    f"gemini_web: __Secure-1PSID not found in {browser} cookies"
                    f"{f' at {self._cookie_file}' if self._cookie_file else ''}. "
                    "Log into https://gemini.google.com/ in that browser first."
                )
            return psid, psidts
        # Let the library try its own fallbacks (env / auto detect). It will
        # raise if nothing works; we surface that as OperatorActionRequired in
        # _ensure_client.
        return None, None

    # ---- Provider protocol -------------------------------------------------

    async def call(self, messages: list[Message], tools: list[ToolDef]) -> CallResponse:
        if tools:
            raise OperatorActionRequired(
                "gemini_web does not support tool/function calling; "
                "configure a different provider for stages that pass tools "
                "(e.g. anthropic, openai, or the native gemini provider)."
            )

        client = await self._ensure_client()
        prompt, files = _flatten_messages_to_prompt(messages)
        cleanup: list[Path] = []
        for part in files:
            cleanup.append(part)

        async def _send() -> str:
            model = _resolve_model(self._model_name)
            try:
                # Use `generate_content(temporary=True)` directly instead of
                # `start_chat().send_message()`:
                #
                #   - `temporary=True` skips writing the turn to the account's
                #     visible chat history. That's the right semantics for a
                #     batch translation pipeline and measurably trims latency
                #     because the server skips index/sidebar bookkeeping.
                #
                #   - We don't need a ChatSession either — every call is a
                #     standalone one-shot turn (the gateway above us serves
                #     the multi-turn history via system+user assembly), so
                #     creating a session per call is just allocation overhead.
                resp = await client.generate_content(
                    prompt=prompt,
                    files=cleanup or None,
                    model=model,
                    temporary=True,
                )
            except Exception as exc:
                if _is_auth_failure(exc):
                    raise OperatorActionRequired(
                        f"gemini_web rejected the request as unauthenticated: {exc}"
                    ) from exc
                if _is_retryable(exc):
                    raise UpstreamUnavailable(f"gemini_web transient failure: {exc}") from exc
                raise
            return resp.text or ""

        try:
            text = await with_retry(
                _send,
                is_retryable=_is_retryable,
                parse_retry_after=lambda _exc: None,
                provider="gemini_web",
            )
        finally:
            for p in cleanup:
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001 — temp cleanup is best-effort
                    pass

        return CallResponse(text=text or None, tool_calls=[])


# ----- helpers --------------------------------------------------------------


def _resolve_model(name: str) -> Any:
    """Map a config model name to a `gemini_webapi.constants.Model` member.

    Special value `"auto"` (and the legacy `"default"`) leaves selection to
    the upstream `UNSPECIFIED` — Gemini picks whatever model the account's
    UI defaults to. All other names must match a canonical model id like
    `gemini-3-flash` or `gemini-3-pro-plus`.
    """
    from gemini_webapi.constants import Model

    if name in ("auto", "default", "", None):
        return Model.UNSPECIFIED
    try:
        return Model.from_name(name)
    except ValueError as exc:
        raise OperatorActionRequired(str(exc)) from exc


def _flatten_messages_to_prompt(messages: list[Message]) -> tuple[str, list[Path]]:
    """Render IR messages into a single prompt + list of temp image files.

    Web Gemini has no role separation, so we:
      - join the SYSTEM message at the top (if present),
      - join prior USER/ASSISTANT turns as `User:`/`Assistant:` blocks,
      - and append the final USER text inline.

    Image parts are extracted into temp files (the library accepts file paths
    only). Caller is responsible for unlinking them after the call.
    """
    parts: list[str] = []
    files: list[Path] = []

    system_text: str | None = None
    for msg in messages:
        if msg.role == Role.SYSTEM and msg.text:
            system_text = msg.text
            break
    if system_text:
        parts.append(system_text.rstrip())

    body_lines: list[str] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue
        if msg.role == Role.TOOL_RESULT:
            # Shouldn't happen (we reject tools), but guard anyway.
            continue
        prefix = "User" if msg.role == Role.USER else "Assistant"
        text_chunks: list[str] = []
        for part in msg.parts:
            if part.text:
                text_chunks.append(part.text)
            if part.image_data_uri:
                tmp = _materialize_image(part.image_data_uri)
                if tmp is not None:
                    files.append(tmp)
                    text_chunks.append(f"[attached: {tmp.name}]")
        if text_chunks:
            body_lines.append(f"{prefix}: " + "\n".join(text_chunks))

    if body_lines:
        parts.append("\n\n".join(body_lines))

    return "\n\n".join(parts).strip(), files


def _materialize_image(data_uri: str) -> Path | None:
    m = _DATA_URI_RE.match(data_uri)
    if not m:
        return None
    mime, b64 = m.group(1), m.group(2)
    ext = {
        "image/jpeg": ".jpg",
        "image/png":  ".png",
        "image/webp": ".webp",
        "image/gif":  ".gif",
    }.get(mime, ".bin")
    fd, tmpname = tempfile.mkstemp(prefix="typoon-gemweb-", suffix=ext)
    try:
        import os as _os

        _os.write(fd, base64.b64decode(b64))
    finally:
        try:
            import os as _os

            _os.close(fd)
        except OSError:
            pass
    return Path(tmpname)
