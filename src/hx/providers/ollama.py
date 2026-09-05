from __future__ import annotations

from typing import Any

from hx.errors import MissingExtra, ProviderError
from hx.types import Completion, Message, StopReason, ToolCall, Usage

try:
    import ollama
except ModuleNotFoundError as e:
    raise MissingExtra("ollama") from e


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str = "gemma4:e2b-mlx",
        max_tokens: int = 4096,
        **client_kw: Any,
    ):
        self._client = ollama.AsyncClient(**client_kw)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages, tools, **kw) -> Completion:
        kwargs: dict[str, Any] = {
            "model": kw.get("model", self._model),
            "messages": self._to_wire(messages),
            "options": {"num_predict": kw.get("max_tokens", self._max_tokens)},
        }
        if tools:
            kwargs["tools"] = [self._tool_to_wire(t) for t in tools]
        try:
            resp = await self._client.chat(**kwargs)
        except Exception as e:
            raise ProviderError(f"ollama: {e}") from e
        return self._from_wire(resp)

    def count_tokens(self, messages) -> int:
        # Aproximação de 4 chars/token. Suficiente para decisão de compactação.
        chars = sum(len(m.content) for m in messages)
        chars += sum(len(str(c.arguments)) for m in messages for c in m.tool_calls)
        return chars // 4

    # ---------- tradução ----------

    def _tool_to_wire(self, t: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }

    def _to_wire(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": m.content,
                        "tool_name": self._tool_name_from_id(m.tool_call_id),
                    }
                )
                continue

            if m.role == "assistant" and m.tool_calls:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments,
                            }
                        }
                        for c in m.tool_calls
                    ],
                }
                out.append(msg)
                continue

            out.append({"role": m.role, "content": m.content})
        return out

    def _from_wire(self, resp) -> Completion:
        msg = resp.message
        text = msg.content or ""
        raw_calls = msg.tool_calls or []
        calls = tuple(
            ToolCall(
                id=f"{i}:{tc.function.name}",
                name=tc.function.name,
                arguments=dict(tc.function.arguments or {}),
            )
            for i, tc in enumerate(raw_calls)
        )
        usage = Usage(
            input_tokens=getattr(resp, "prompt_eval_count", None) or 0,
            output_tokens=getattr(resp, "eval_count", None) or 0,
        )
        return Completion(
            message=Message("assistant", text, tool_calls=calls),
            usage=usage,
            stop_reason=self._stop_reason(resp, calls),
            raw=resp,
        )

    def _stop_reason(self, resp, calls: tuple[ToolCall, ...]) -> StopReason:
        if calls:
            return "tool_use"
        reason = getattr(resp, "done_reason", None)
        if reason == "length":
            return "max_tokens"
        if reason == "stop":
            return "end_turn"
        return "stop"

    @staticmethod
    def _tool_name_from_id(tool_call_id: str | None) -> str:
        if not tool_call_id:
            return ""
        if ":" in tool_call_id:
            return tool_call_id.split(":", 1)[1]
        return tool_call_id
