"""VisionRuntime adapter — bridges Config/Paths into the vision runtime.

Thin factory only. Callers receive a `vision.VisionRuntime` directly;
this adapter exists to (a) read the typed Config/Paths, (b) resolve the
spec preset + overrides, and (c) own the shared ModelHub.

The returned `vision.VisionRuntime` is what scan/render stages consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typoon.models import ModelHub
from typoon.vision.pipeline import VisionPipelineSpec
from typoon.vision.runtime import VisionRuntime, build_vision_runtime


__all__ = ["VisionRuntimeAdapter", "build_from_config"]


@dataclass(slots=True)
class VisionRuntimeAdapter:
    """Pair of (vision runtime, model hub) constructed from Config.

    Worker code holds this via `ctx.runtime`. Stages use `.runtime` for
    pipeline calls and `.hub` for ad-hoc model lookups.
    """

    runtime: VisionRuntime
    hub:     ModelHub

    @staticmethod
    def from_config(config=None, paths=None, *, source_lang: str | None = None):
        return build_from_config(config, paths, source_lang=source_lang)


def build_from_config(
    config=None, paths=None, *, source_lang: str | None = None,
) -> tuple[VisionRuntimeAdapter, object, object]:
    from typoon.config import load_config
    if config is None or paths is None:
        config, paths = load_config()
    hub = ModelHub(Path(config.models_dir))
    spec = _resolve_spec(config)
    runtime = build_vision_runtime(
        spec,
        models_dir=hub.dir,
        source_lang=source_lang,
        lens_endpoint=config.lens_endpoint or None,
    )
    return VisionRuntimeAdapter(runtime=runtime, hub=hub), config, paths


def _resolve_spec(config) -> VisionPipelineSpec:
    """Read `config.vision` and resolve to a VisionPipelineSpec.

    Schema (`config.toml`):
        [vision]
        preset = "lens"           # PRESETS key
        # Optional per-stage overrides on top of the preset:
        # recognizer = "apple_vision"
        # page_concurrency = 8

    If `config.vision` is absent, defaults to the `lens` preset.
    """
    raw = getattr(config, "vision", None)
    if raw is None:
        return VisionPipelineSpec.preset("lens")

    if isinstance(raw, str):
        return VisionPipelineSpec.preset(raw)

    preset_name = (
        raw.get("preset") if isinstance(raw, dict)
        else getattr(raw, "preset", "lens")
    ) or "lens"
    base = VisionPipelineSpec.preset(preset_name)

    overrides: dict[str, object] = {}
    for field_name in (
        "detector", "grouper", "recognizer", "eraser",
        "page_concurrency", "detect_concurrency", "erase_concurrency",
    ):

        value = (
            raw.get(field_name) if isinstance(raw, dict)
            else getattr(raw, field_name, None)
        )
        if value is not None:
            overrides[field_name] = value
    return base.with_overrides(**overrides) if overrides else base
