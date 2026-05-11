"""Anthropic Claude provider."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from tau.config import TauConfig
from tau.core.types import (
    AgentConfig,
    Message,
    ProviderResponse,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
}


class AnthropicProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is required: pip install anthropic") from exc

        self._agent_config = agent_config
        self._model = agent_config.model
        self._last_response_headers: dict[str, str] = {}
        self._client = Anthropic(
            api_key=config.anthropic.api_key or None,
            base_url=(config.anthropic.base_url or None),
        )

    @property
    def last_response_headers(self) -> dict[str, str]:
        return self._last_response_headers

    @contextlib.contextmanager
    def swap_model(self, model: str):
        old_model = self._model
        self._model = model
        try:
            yield
        finally:
            self._model = old_model

    @property
    def name(self) -> str:
        return "anthropic"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse:
        del stream  # Minimal v1: use blocking call for reliability.
        system_text, anth_messages = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max(256, min(8192, self._agent_config.max_tokens)),
            "messages": anth_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]

        logger.debug(
            "Anthropic request: model=%s, messages=%d, tools=%d",
            self._model,
            len(anth_messages),
            len(tools),
        )
        response = self._client.messages.create(**kwargs)
        return _parse_anthropic_response(response)


def _to_anthropic_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.to_json_schema(),
    }


def _image_block(path: str) -> dict[str, Any] | None:
    p = Path(path)
    try:
        mime_type, _ = mimetypes.guess_type(str(p))
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encoded,
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load image for Anthropic %s: %s", path, exc)
        return None


def _to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role == "user":
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            if m.images:
                for img_path in m.images:
                    ib = _image_block(img_path)
                    if ib is not None:
                        parts.append(ib)
            if not parts:
                parts = [{"type": "text", "text": ""}]
            out.append({"role": "user", "content": parts})
            continue

        if m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments or {},
                        }
                    )
            if not parts:
                parts = [{"type": "text", "text": ""}]
            out.append({"role": "assistant", "content": parts})
            continue

        if m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id or "",
                            "content": m.content or "",
                        }
                    ],
                }
            )

    return "\n\n".join(system_parts), out


def _parse_anthropic_response(response: Any) -> ProviderResponse:
    blocks = getattr(response, "content", []) or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for b in blocks:
        b_type = getattr(b, "type", "")
        if b_type == "text":
            text = getattr(b, "text", None)
            if text:
                text_parts.append(text)
        elif b_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(b, "id", ""),
                    name=getattr(b, "name", ""),
                    arguments=getattr(b, "input", {}) or {},
                )
            )

    usage_obj = getattr(response, "usage", None)
    usage = TokenUsage(
        input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
    )
    stop_raw = str(getattr(response, "stop_reason", "end_turn") or "end_turn")
    stop_reason = _STOP_REASON_MAP.get(stop_raw, "end_turn")
    if tool_calls:
        stop_reason = "tool_use"

    return ProviderResponse(
        content="\n".join(text_parts) or None,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )

