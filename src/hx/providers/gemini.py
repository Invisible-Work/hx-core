from __future__ import annotations

from typing import Any

from hx.errors import MissingExtra, ProviderError
from hx.types import Completion, Message, StopReason, ToolCall, Usage

try:
    from google import genai
    from google.genai import types
except ModuleNotFoundError as e:
    raise MissingExtra("gemini") from e

_STOP: dict[str, StopReason] = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
}

# Prefix for ids we invent when Gemini omits FunctionCall.id (older models).
_SYNTH_PREFIX = "hx_"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        max_tokens: int = 4096,
        **client_kw: Any,
    ):
        # Client exige API key; lazy para isinstance()/construção sem env.
        self._client_kw = client_kw
        self._client: Any = None
        self._model = model
        self._max_tokens = max_tokens

    def _aio(self):
        if self._client is None:
            self._client = genai.Client(**self._client_kw).aio
        return self._client

    async def complete(self, messages, tools, **kw) -> Completion:
        system, convo = self._split_system(messages)
        config_kw: dict[str, Any] = {
            "max_output_tokens": kw.get("max_tokens", self._max_tokens),
            # hx executa tools; AFC em generate_content gera warning e é indesejado.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system:
            config_kw["system_instruction"] = system
        if tools:
            decls = [self._tool_to_wire(t) for t in tools]
            config_kw["tools"] = [types.Tool(function_declarations=decls)]
        try:
            resp = await self._aio().models.generate_content(
                model=kw.get("model", self._model),
                contents=self._to_wire(convo),
                config=types.GenerateContentConfig(**config_kw),
            )
        except Exception as e:
            raise ProviderError(f"gemini: {e}") from e
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

    def _tool_to_wire(self, t: dict) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters_json_schema=t["input_schema"],
        )

    def _to_wire(self, convo: list[Message]) -> list[types.Content]:
        out: list[types.Content] = []
        id_to_name: dict[str, str] = {}

        for m in convo:
            if m.role == "tool":
                name = id_to_name.get(m.tool_call_id or "", "")
                fr_kw: dict[str, Any] = {
                    "name": name,
                    "response": {"result": m.content},
                }
                wire_id = self._wire_call_id(m.tool_call_id)
                if wire_id:
                    fr_kw["id"] = wire_id
                part = types.Part(
                    function_response=types.FunctionResponse(**fr_kw)
                )
                if (
                    out
                    and out[-1].role == "tool"
                    and out[-1].parts is not None
                ):
                    out[-1].parts.append(part)
                else:
                    out.append(types.Content(role="tool", parts=[part]))
                continue

            if m.role == "assistant" and m.tool_calls:
                parts: list[types.Part] = []
                if m.content:
                    parts.append(types.Part.from_text(text=m.content))
                for c in m.tool_calls:
                    id_to_name[c.id] = c.name
                    fc_kw: dict[str, Any] = {
                        "name": c.name,
                        "args": c.arguments,
                    }
                    wire_id = self._wire_call_id(c.id)
                    if wire_id:
                        fc_kw["id"] = wire_id
                    parts.append(
                        types.Part(function_call=types.FunctionCall(**fc_kw))
                    )
                out.append(types.Content(role="model", parts=parts))
                continue

            role = "model" if m.role == "assistant" else m.role
            out.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m.content)],
                )
            )
        return out

    def _from_wire(self, resp) -> Completion:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        finish = None

        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            finish = getattr(cand, "finish_reason", None)
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or [] if content else []
            for i, part in enumerate(parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None and fc.name:
                    call_id = fc.id or f"{_SYNTH_PREFIX}{fc.name}_{i}"
                    args = dict(fc.args) if fc.args else {}
                    calls.append(
                        ToolCall(id=call_id, name=fc.name, arguments=args)
                    )

        usage_raw = getattr(resp, "usage_metadata", None)
        if usage_raw is not None:
            usage = Usage(
                input_tokens=getattr(usage_raw, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage_raw, "candidates_token_count", 0)
                or 0,
                cached_tokens=getattr(
                    usage_raw, "cached_content_token_count", 0
                )
                or 0,
            )
        else:
            usage = Usage()

        finish_key = getattr(finish, "name", None) or str(finish or "")
        stop: StopReason = "tool_use" if calls else _STOP.get(finish_key, "stop")

        return Completion(
            message=Message("assistant", "".join(text_parts), tool_calls=tuple(calls)),
            usage=usage,
            stop_reason=stop,
            raw=resp,
        )

    @staticmethod
    def _wire_call_id(hx_id: str | None) -> str | None:
        """Só reenvia id nativo do Gemini; ids sintéticos hx_* ficam fora do wire."""
        if hx_id and not hx_id.startswith(_SYNTH_PREFIX):
            return hx_id
        return None
