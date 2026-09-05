from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from hx.types import Completion, Message, ToolCall

if TYPE_CHECKING:
    from hx.core import AgentState


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(
        self, messages: list[Message], tools: list[dict], **kw: Any
    ) -> Completion: ...

    def count_tokens(self, messages: list[Message]) -> int: ...


@runtime_checkable
class MemoryDriver(Protocol):
    async def append(self, session_id: str, msgs: list[Message]) -> None: ...
    async def load(self, session_id: str) -> list[Message]: ...
    async def recall(
        self, session_id: str, query: str, k: int = 5
    ) -> list[Message]: ...


@runtime_checkable
class ContextStrategy(Protocol):
    async def build(self, state: "AgentState") -> list[Message]: ...


@runtime_checkable
class Sandbox(Protocol):
    async def run(self, cmd: str, timeout: int = 30) -> tuple[int, str, str]: ...


@runtime_checkable
class Hook(Protocol):
    async def on_turn_start(self, state: "AgentState") -> None: ...
    async def on_tool_call(self, call: ToolCall) -> None: ...
    async def on_tool_result(self, call: ToolCall, result: Any) -> None: ...
    async def on_turn_end(
        self, state: "AgentState", completion: Completion
    ) -> None: ...