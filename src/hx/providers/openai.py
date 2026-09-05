from __future__ import annotations

import json
from typing import Any

from hx.errors import MissingExtra, ProviderError
from hx.types import Completion, Message, ToolCall, Usage

try:
    import openai
except ModuleNotFoundError as e:
    raise MissingExtra("openai") from e


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        max_tokens: int = 4096,
        **client_kw: Any,
    ):
        self._client = openai.AsyncOpenAI(**client_kw)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages, tools, **kw) -> Completion:
        system, convo = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": kw.get("model", self._model),
            "input": self._to_wire(convo),
            "max_output_tokens": kw.get("max_tokens", self._max_tokens),
        }
        if system:
            kwargs["instructions"] = system
        if tools:
            kwargs["tools"] = [self._tool_to_wire(t) for t in tools]
        try:
            resp = await self._client.responses.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"openai: {e}") from e
        return self._from_wire(resp)

    def count_tokens(self, messages) -> int:
        # Aproximação de 4 chars/token. Suficiente para decisão de compactação.
        chars = sum(len(m.content) for m in messages)
        chars += sum(len(str(c.arguments)) for m in messages for c in m.tool_calls)
        return chars // 4

    # ---------- tradução ----------

    def _split_system(self, messages) -> tuple[str, list[Message]]:
        sys_parts = [m.content for m in messages if m.role == "system"]
        convo = [m for m in messages if m.role != "system"]
        return "\n\n".join(sys_parts), convo

    def _tool_to_wire(self, t: dict) -> dict:
        return {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["input_schema"],
        }

    def _to_wire(self, convo: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in convo:
            if m.role == "tool":
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.tool_call_id,
                        "output": m.content,
                    }
                )
                continue

            if m.role == "assistant" and m.tool_calls:
                if m.content:
                    out.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": m.content,
                        }
                    )
                for c in m.tool_calls:
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": c.id,
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        }
                    )
                continue

            out.append(
                {
                    "type": "message",
                    "role": m.role,
                    "content": m.content,
                }
            )
        return out

    def _from_wire(self, resp) -> Completion:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for item in resp.output or []:
            if item.type == "message":
                for part in item.content or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(part.text)
                    elif isinstance(part, str):
                        text_parts.append(part)
            elif item.type == "function_call":
                args = item.arguments
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                calls.append(
                    ToolCall(id=item.call_id, name=item.name, arguments=args)
                )

        text = "".join(text_parts) or (getattr(resp, "output_text", None) or "")
        usage_raw = resp.usage
        cached = 0
        if usage_raw is not None:
            details = getattr(usage_raw, "input_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            usage = Usage(
                input_tokens=getattr(usage_raw, "input_tokens", 0) or 0,
                output_tokens=getattr(usage_raw, "output_tokens", 0) or 0,
                cached_tokens=cached,
            )
        else:
            usage = Usage()

        return Completion(
            message=Message("assistant", text, tool_calls=tuple(calls)),
            usage=usage,
            stop_reason=self._stop_reason(resp, calls),
            raw=resp,
        )

    def _stop_reason(self, resp, calls: list[ToolCall]) -> str:
        if calls:
            return "tool_use"
        if getattr(resp, "status", None) == "incomplete":
            details = getattr(resp, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            if reason == "max_output_tokens":
                return "max_tokens"
            return "stop"
        if getattr(resp, "status", None) == "completed":
            return "end_turn"
        return "stop"
