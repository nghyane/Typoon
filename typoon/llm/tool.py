"""Tool — unified definition + handler binding for LLM tool calls."""

from __future__ import annotations

import textwrap
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints
import inspect

from pydantic import BaseModel

from .ir import ToolDef, ToolResponse, ToolCallMsg


class Tool:
    """Binds a ToolDef, its Pydantic args model, and an async handler.

    Construct via ``Tool.define()``.
    Dispatch via ``await tool.call(raw_json)``.
    Pass ``tool.definition`` to the provider.
    """

    __slots__ = ("definition", "_args_model", "_handler")

    def __init__(
        self,
        definition: ToolDef,
        args_model: type[BaseModel],
        handler: Callable[..., Awaitable[ToolResponse]],
    ) -> None:
        self.definition = definition
        self._args_model = args_model
        self._handler = handler

    @staticmethod
    def define(
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[..., Awaitable[ToolResponse]],
    ) -> "Tool":
        """Create a Tool from explicit parts."""
        schema = _build_schema(args_model)
        return Tool(ToolDef(name, description, schema), args_model, handler)

    async def call(self, tc: ToolCallMsg) -> ToolResponse:
        """Parse args and invoke handler. Returns error ToolResponse on bad args."""
        try:
            args = self._args_model.model_validate_json(tc.arguments or "{}")
        except Exception as e:
            return ToolResponse(f"Invalid arguments for {tc.name}: {e}")
        return await self._handler(args)


def tool(fn: Callable | None = None) -> Any:
    """Decorator: build a Tool factory from a Pydantic-typed async function.

    Usage::

        @tool
        async def my_tool(args: MyArgs, ctx: SomeCtx) -> ToolResponse:
            ...

        t = my_tool(ctx=some_ctx)  # returns Tool bound to ctx

    The decorated function's first Pydantic-typed parameter is the args model.
    Remaining parameters become the factory closure kwargs.
    """
    def decorator(f: Callable) -> Callable[..., Tool]:
        hints = get_type_hints(f)
        params = list(inspect.signature(f).parameters.values())

        args_model: type[BaseModel] | None = None
        args_param: str | None = None
        extra_params: list[str] = []
        for p in params:
            hint = hints.get(p.name)
            if args_model is None and hint is not None and isinstance(hint, type) and issubclass(hint, BaseModel):
                args_model = hint
                args_param = p.name
            else:
                extra_params.append(p.name)

        if args_model is None:
            raise TypeError(f"@tool function {f.__name__} must have a Pydantic BaseModel parameter")

        schema = _build_schema(args_model)
        doc = textwrap.dedent(f.__doc__ or "").strip()
        tdef = ToolDef(f.__name__, doc, schema)

        def factory(**kwargs: Any) -> Tool:
            async def handler(args: BaseModel) -> ToolResponse:
                return await f(**{args_param: args, **kwargs})
            return Tool(tdef, args_model, handler)

        factory.__name__ = f.__name__
        factory.__doc__ = doc
        factory._tool_def = tdef       # expose for introspection
        factory._args_model = args_model
        return factory

    if fn is not None:
        return decorator(fn)
    return decorator


# ── Schema helpers ────────────────────────────────────────────────────


def _build_schema(model: type[BaseModel]) -> dict:
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
