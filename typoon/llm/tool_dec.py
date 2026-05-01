"""@tool decorator — auto-generate ToolDef from Pydantic model + docstring."""

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


def tool() -> Any:
    """Decorator: @tool() — generate ToolDef from Pydantic model + docstring."""

    def decorator(fn: Any) -> ToolFunc:
        hints = get_type_hints(fn)
        params = list(inspect.signature(fn).parameters.values())
        args_model: type[BaseModel] | None = None
        for p in params:
            hint = hints.get(p.name)
            if hint is not None and isinstance(hint, type) and issubclass(hint, BaseModel):
                args_model = hint
                break

        if args_model is None:
            raise TypeError(f"@tool function {fn.__name__} must have a Pydantic BaseModel parameter")

        schema = _build_schema(args_model)
        doc = textwrap.dedent(fn.__doc__ or "").strip()
        return ToolFunc(fn, ToolDef(fn.__name__, doc, schema), args_model)

    return decorator


def _build_schema(model: type[BaseModel]) -> dict:
    """Build JSON schema from Pydantic model: resolve $ref, strip title."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}
    resolved = _resolve(schema, defs) if defs else schema
    _strip_keys(resolved, {"title"})
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
    if isinstance(node, dict):
        for k in keys:
            node.pop(k, None)
        for v in node.values():
            _strip_keys(v, keys)
    elif isinstance(node, list):
        for item in node:
            _strip_keys(item, keys)
