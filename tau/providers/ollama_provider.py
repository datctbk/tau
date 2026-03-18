"""Ollama provider — uses the Ollama REST API via httpx."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from typing import Any

import httpx

from tau.config import TauConfig
from tau.core.types import (
    AgentConfig,
    Message,
    ProviderResponse,
    StopReason,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


class OllamaProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        self._model = agent_config.model
        self._base_url = config.ollama.base_url.rstrip("/")
        self._client = httpx.Client(timeout=120)

    @property
    def name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse | Generator:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]

        logger.debug("Ollama request: model=%s, messages=%d, stream=%s",
                     self._model, len(messages), stream)

        if stream:
            return self._chat_stream(payload)
        return self._chat_blocking(payload)

    # ------------------------------------------------------------------

    def _chat_blocking(self, payload: dict[str, Any]) -> ProviderResponse:
        resp = self._client.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return _parse_ollama_response(data)

    def _chat_stream(self, payload: dict[str, Any]) -> Generator:
        content_parts: list[str] = []
        tool_calls_raw: list[dict] = []
        final_data: dict[str, Any] = {}

        with self._client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})

                if msg.get("tool_calls"):
                    tool_calls_raw.extend(msg["tool_calls"])

                # Thinking tokens — yield with is_thinking=True, never buffer
                thinking = msg.get("thinking", "")
                if thinking:
                    yield TextDelta(text=thinking, is_thinking=True)  # type: ignore[misc]

                # Visible content
                delta = msg.get("content", "")
                if delta:
                    content_parts.append(delta)
                    yield TextDelta(text=delta)  # type: ignore[misc]

                if chunk.get("done"):
                    final_data = chunk

        final_data.setdefault("message", {})
        final_data["message"]["content"] = "".join(content_parts)
        if tool_calls_raw:
            final_data["message"]["tool_calls"] = tool_calls_raw

        yield _parse_ollama_response(final_data)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Shared response parser
# ---------------------------------------------------------------------------

def _parse_ollama_response(data: dict[str, Any]) -> ProviderResponse:
    msg = data.get("message", {})
    finish = data.get("done_reason", "stop")
    stop_reason: StopReason = _STOP_REASON_MAP.get(finish, "end_turn")

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        tool_calls.append(ToolCall(
            id=str(uuid.uuid4()),
            name=fn.get("name", ""),
            arguments=args,
        ))
    if tool_calls:
        stop_reason = "tool_use"

    usage = TokenUsage(
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )
    return ProviderResponse(
        content=msg.get("content") or None,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_ollama_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content or ""}


def _to_ollama_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_json_schema(),
        },
    }
