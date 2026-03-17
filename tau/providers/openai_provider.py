"""OpenAI (and OpenAI-compatible) provider."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

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
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "error",
}


class OpenAIProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai") from exc

        self._model = agent_config.model
        self._client = OpenAI(
            api_key=config.openai.api_key or None,
            base_url=config.openai.base_url,
        )

    @property
    def name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse:
        oai_messages = [_to_oai_message(m) for m in messages]
        oai_tools = [_to_oai_tool(t) for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "stream": stream,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        logger.debug("OpenAI request: model=%s, messages=%d, stream=%s",
                     self._model, len(oai_messages), stream)

        if stream:
            return self._chat_stream(kwargs)
        return self._chat_blocking(kwargs)

    # ------------------------------------------------------------------

    def _chat_blocking(self, kwargs: dict[str, Any]) -> ProviderResponse:
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason or "stop"
        stop_reason: StopReason = _STOP_REASON_MAP.get(finish, "end_turn")

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                ))

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return ProviderResponse(content=msg.content, tool_calls=tool_calls,
                                stop_reason=stop_reason, usage=usage)

    def _chat_stream(self, kwargs: dict[str, Any]) -> ProviderResponse:
        # Accumulators
        content_parts: list[str] = []
        # tool call accumulators: index → {id, name, arguments}
        tc_acc: dict[int, dict[str, Any]] = {}
        stop_reason: StopReason = "end_turn"
        usage = TokenUsage()

        with self._client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                # Usage chunk (last chunk when stream_options.include_usage=True)
                if chunk.usage:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")

                delta = choice.delta

                # Text delta — yield via the generator protocol
                if delta.content:
                    content_parts.append(delta.content)
                    yield TextDelta(text=delta.content)  # type: ignore[misc]

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_acc:
                            tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_acc[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = [
            ToolCall(
                id=v["id"],
                name=v["name"],
                arguments=json.loads(v["arguments"] or "{}"),
            )
            for v in tc_acc.values()
        ]
        if tool_calls:
            stop_reason = "tool_use"

        yield ProviderResponse(  # type: ignore[misc]
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _to_oai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content or ""}
def _to_oai_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_json_schema(),
        },
    }
