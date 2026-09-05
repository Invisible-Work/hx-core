from __future__ import annotations

from typing import Any

from hx.errors import MissingExtra, ProviderError
from hx.types import Completion, Message, ToolCall, Usage

try:
    import anthropic
except ModuleNotFoundError as e:
    raise MissingExtra("anthropic") from e

_STOP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop",
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        **client_kw: Any,
    ):
        self._client = anthropic.AsyncAnthropic(**client_kw)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages, tools, **kw) -> Completion:
        system, convo = self._split_system(messages)
        try:
            resp = await self._client.messages.create(
                model=kw.get("model", self._model),
                max_tokens=kw.get("max_tokens", self._max_tokens),
                system=system or anthropic.NOT_GIVEN,
                messages=self._to_wire(convo),
                tools=[self._tool_to_wire(t) for t in tools],
            )
        except Exception as e:
            raise ProviderError(f"anthropic: {e}") from e
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
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["input_schema"],
        }

    def _to_wire(self, convo: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in convo:
            if m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                # resultados consecutivos são agrupados numa única mensagem
                if out and out[-1]["role"] == "user" and isinstance(
                    out[-1]["content"], list
                ) and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue

            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    }
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks})
                continue

            out.append({"role": m.role, "content": m.content})
        return out

    def _from_wire(self, resp) -> Completion:
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = tuple(
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content
            if b.type == "tool_use"
        )
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cached_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )
        return Completion(
            message=Message("assistant", text, tool_calls=calls),
            usage=usage,
            stop_reason=_STOP.get(resp.stop_reason, "stop"),
            raw=resp,
        )