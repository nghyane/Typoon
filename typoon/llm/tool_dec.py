"""@tool decorator — auto-generate ToolDef from Pydantic model + docstring.

Usage:
    class TranslateArgs(BaseModel):
        translations: list[Item]

    @tool(strict=True)
    async def translate(args: TranslateArgs) -> str:
        '''Submit translations for one or more bubbles.'''
        ...

    translate.definition   # → ToolDef
    translate.args_model   # → TranslateArgs
    await translate(args)  # → calls the function
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, get_type_hints

from pydantic import BaseModel

from .ir import ToolDef


class ToolFunc:
    """Wrapper that holds both the callable and its ToolDef."""

    __slots__ = ("fn", "definition", "args_model")

    def __init__(self, fn: Any, definition: ToolDef, args_model: type[BaseModel]) -> None:
        self.fn = fn
        self.definition = definition
        self.args_model = args_model

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.fn(*args, **kwargs)

    def parse_args(self, raw_json: str) -> BaseModel:
        return self.args_model.model_validate_json(raw_json)


def tool(*, strict: bool = False) -> Any:
    """Decorator factory: @tool(strict=True)"""

    def decorator(fn: Any) -> ToolFunc:
        hints = get_type_hints(fn)

        # Find the Pydantic model from the first parameter
        params = list(inspect.signature(fn).parameters.values())
        args_model: type[BaseModel] | None = None
        for p in params:
            hint = hints.get(p.name)
            if hint is not None and isinstance(hint, type) and issubclass(hint, BaseModel):
                args_model = hint
                break

        if args_model is None:
            raise TypeError(f"@tool function {fn.__name__} must have a Pydantic BaseModel parameter")

        schema = _clean_schema(args_model.model_json_schema(), strict=strict)
        doc = textwrap.dedent(fn.__doc__ or "").strip()

        return ToolFunc(fn, ToolDef(fn.__name__, doc, schema, strict), args_model)

    return decorator


def _clean_schema(schema: dict, strict: bool = False) -> dict:
    """Resolve $ref, inline $defs, strip title, add additionalProperties for strict mode."""
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
    resolved = _resolve(schema, defs) if defs else schema
    _strip_keys(resolved, {"title"})
    if strict:
        _add_additional_properties_false(resolved)
        _force_all_required(resolved)
    return resolved


def _resolve(node: Any, defs: dict) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return _resolve(dict(defs.get(ref_name, {})), defs)
        return {k: _resolve(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    return node


def _strip_keys(node: Any, keys: set[str]) -> None:
    """Remove keys in-place, recursively."""
    if isinstance(node, dict):
        for k in keys:
            node.pop(k, None)
        for v in node.values():
            _strip_keys(v, keys)
    elif isinstance(node, list):
        for item in node:
            _strip_keys(item, keys)


def _add_additional_properties_false(node: Any) -> None:
    """Add additionalProperties: false to all object schemas (required by OpenAI strict mode)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node.setdefault("additionalProperties", False)
        for v in node.values():
            _add_additional_properties_false(v)
    elif isinstance(node, list):
        for item in node:
            _add_additional_properties_false(item)


def _force_all_required(node: Any) -> None:
    """OpenAI strict mode: every key in `properties` must appear in `required`.

    Pydantic omits fields with defaults from `required`; strict mode rejects
    that (error: 'Missing <field>' in tools[N].function.parameters). Override
    by setting required = list(properties.keys()) on every object schema.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _force_all_required(v)
    elif isinstance(node, list):
        for item in node:
            _force_all_required(item)
