"""Shared types for tau."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None   # for role="tool" results
    tool_calls: list[ToolCall] | None = None  # for role="assistant" with calls
    name: str | None = None           # tool name (some providers require it)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        tool_calls = None
        if "tool_calls" in d:
            tool_calls = [ToolCall.from_dict(tc) for tc in d["tool_calls"]]
        return cls(
            role=d["role"],
            content=d.get("content") or "",
            tool_call_id=d.get("tool_call_id"),
            tool_calls=tool_calls,
            name=d.get("name"),
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass
class ToolParameter:
    """JSON Schema fragment describing one parameter."""
    type: str
    description: str
    enum: list[str] | None = None
    required: bool = True


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, ToolParameter]  # param_name → ToolParameter
    handler: Callable[..., Any]

    def to_json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema object for the function parameters."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in self.parameters.items():
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[pname] = prop
            if p.required:
                required.append(pname)
        return {
            "type": "object",
            "properties": props,
            "required": required,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(id=d["id"], name=d["name"], arguments=d.get("arguments", {}))


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderResponse:
    content: str | None
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: TokenUsage = field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    max_tokens: int = 8192
    max_turns: int = 20
    system_prompt: str = (
        "You are tau, a minimal CLI coding agent. "
        "Use the available tools to help the user with coding tasks. "
        "Be concise and precise."
    )
    trim_strategy: Literal["sliding_window", "summarise"] = "sliding_window"
    workspace_root: str = "."


# ---------------------------------------------------------------------------
# Agent events (streamed from agent loop → CLI renderer)
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    text: str


@dataclass
class TextDelta:
    """A single streaming token/chunk from the LLM (not yet complete)."""
    text: str


@dataclass
class ToolCallEvent:
    call: ToolCall


@dataclass
class ToolResultEvent:
    result: ToolResult


@dataclass
class TurnComplete:
    usage: TokenUsage


@dataclass
class ErrorEvent:
    message: str


Event = TextChunk | TextDelta | ToolCallEvent | ToolResultEvent | TurnComplete | ErrorEvent
