from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from hx.errors import PermissionDenied, ToolNotFound
from hx.types import ToolCall

Risk = Literal["read_only", "mutating", "destructive"]


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        node.pop("title", None)
        return {k: _strip_titles(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _args_model(fn: Callable) -> type[BaseModel]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ann = hints.get(name, Any)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (ann, default)
    return create_model(f"{fn.__name__}_Args", **fields)


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable
    model: type[BaseModel]
    schema: dict
    risk: Risk
    description: str
    is_async: bool

# src/hx/tools.py (continuação)
class ToolRegistry:
    def __init__(self, allow: tuple[Risk, ...] = ("read_only", "mutating", "destructive")):
        self._specs: dict[str, ToolSpec] = {}
        self._allow = allow

    def tool(self, *, risk: Risk = "read_only"):
        """Decorador. Deriva o schema dos type hints e a descrição do docstring."""
        def deco(fn: Callable) -> Callable:
            model = _args_model(fn)
            schema = _strip_titles(model.model_json_schema())
            schema.setdefault("additionalProperties", False)
            self._specs[fn.__name__] = ToolSpec(
                fn=fn,
                model=model,
                schema=schema,
                risk=risk,
                description=(inspect.getdoc(fn) or "").strip(),
                is_async=inspect.iscoroutinefunction(fn),
            )
            return fn
        return deco

    def schemas(self) -> list[dict]:
        """Formato neutro. Cada Provider traduz para o seu."""
        return [
            {"name": n, "description": s.description, "input_schema": s.schema}
            for n, s in self._specs.items()
            if s.risk in self._allow
        ]

    def filtered(self, allow: tuple[Risk, ...]) -> "ToolRegistry":
        novo = ToolRegistry(allow=allow)
        novo._specs = dict(self._specs)      # não muta o original
        return novo

    async def invoke(self, call: ToolCall) -> Any:
        spec = self._specs.get(call.name)
        if spec is None:
            raise ToolNotFound(f"ferramenta desconhecida: {call.name}")
        if spec.risk not in self._allow:
            raise PermissionDenied(
                f"'{call.name}' exige risco '{spec.risk}', permitido: {self._allow}"
            )
        try:
            args = spec.model(**call.arguments)
        except ValidationError as e:
            raise ValueError(f"argumentos inválidos para {call.name}: {e}") from e

        kwargs = args.model_dump()
        if spec.is_async:
            return await spec.fn(**kwargs)
        return await asyncio.to_thread(spec.fn, **kwargs)