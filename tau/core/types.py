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
    images: list[str] | None = None   # list of absolute file paths to attached images

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name:
            d["name"] = self.name
        if self.images:
            d["images"] = self.images
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
            images=d.get("images"),
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
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


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
    thinking_level: str = "off"
    max_tokens: int = 8192
    max_turns: int = 20
    system_prompt: str = (
        "You are tau, a minimal CLI coding agent. "
        "Use the available tools to help the user with coding tasks. "
        "Be concise and precise."
    )
    trim_strategy: Literal["sliding_window", "summarise"] = "sliding_window"
    workspace_root: str = "."
    # --- auto-compaction ---
    compaction_enabled: bool = True
    compaction_threshold: float = 0.80  # compact when context reaches 80% of max_tokens
    # --- auto-retry ---
    retry_enabled: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 2.0   # seconds; doubles each attempt

# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
@dataclass
class CompactionEntry:
    """Stored in the session to mark a compaction point."""
    summary: str
    tokens_before: int
    timestamp: str

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
@dataclass
class RetryEvent:
    """Emitted when the agent is about to retry after a retryable error."""
    attempt: int          # 1-based current attempt number
    max_attempts: int
    delay: float          # seconds the agent will wait before retrying
    error: str            # the error message that triggered the retry

# ---------------------------------------------------------------------------
# Session branching
# ---------------------------------------------------------------------------
@dataclass
class ForkInfo:
    """Metadata about one fork point (a user message in a session)."""
    index: int          # 0-based position in session.messages
    content: str        # text preview of the user message

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
    is_thinking: bool = False


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


@dataclass
class CompactionEvent:
    """Emitted when auto-compaction runs (start or end)."""
    stage: Literal["start", "end"]
    tokens_before: int = 0
    tokens_after: int = 0
    summary: str = ""
    error: str = ""

@dataclass
class SteerEvent:
    """Emitted when the user steers mid-stream (current response discarded)."""
    new_input: str          # the injected user message
    discarded_tokens: int   # how many tokens of the interrupted response were dropped

# ---------------------------------------------------------------------------
# Extension system
# ---------------------------------------------------------------------------

@dataclass
class ExtensionManifest:
    """Static metadata declared by every Extension subclass."""
    name: str                           # unique snake_case identifier
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    system_prompt_fragment: str = ""    # appended to the agent system prompt


@dataclass
class SlashCommand:
    """A /command registered by an extension."""
    name: str                   # without the leading slash, e.g. "fmt"
    description: str = ""
    usage: str = ""             # e.g. "/fmt [file]"


@dataclass
class ExtensionLoadError:
    """Emitted (as an Event) when an extension fails to load at startup."""
    extension_name: str
    error: str


Event = (
    TextChunk
    | TextDelta
    | ToolCallEvent
    | ToolResultEvent
    | TurnComplete
    | ErrorEvent
    | CompactionEvent
    | RetryEvent
    | SteerEvent
    | ExtensionLoadError
)
