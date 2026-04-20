"""Ollama provider — uses the Ollama REST API via httpx."""

from __future__ import annotations

import json
import logging
import re
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
        timeout_s = max(5.0, float(config.ollama.timeout_seconds))
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s))

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
        buffer = ""
        in_tool_call = False

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

                # Visible content — suppress <tool_call> blocks from display
                delta = msg.get("content", "")
                if delta:
                    buffer += delta

                    # Drain normal content before any <tool_call> opening tag
                    while True:
                        if in_tool_call:
                            if "</tool_call>" in buffer:
                                _, _, buffer = buffer.partition("</tool_call>")
                                in_tool_call = False
                            else:
                                break  # still accumulating inside tag
                        else:
                            if "<tool_call>" in buffer:
                                before, _, after = buffer.partition("<tool_call>")
                                if before:
                                    content_parts.append(before)
                                    yield TextDelta(text=before)  # type: ignore[misc]
                                buffer = "<tool_call>" + after
                                in_tool_call = True
                            else:
                                # Emit everything except a possible partial opening tag
                                safe, _, partial = buffer.rpartition("\n")
                                emit = (safe + "\n") if safe else ""
                                # Hold back content that might be start of <tool_call>
                                if "<" in partial:
                                    emit += ""
                                    buffer = partial
                                else:
                                    emit += partial
                                    buffer = ""
                                if emit:
                                    content_parts.append(emit)
                                    yield TextDelta(text=emit)  # type: ignore[misc]
                                break

                if chunk.get("done"):
                    final_data = chunk

        # Flush remaining non-tool-call buffer
        if buffer and not in_tool_call:
            content_parts.append(buffer)

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

    # Fallback: parse <tool_call> blocks from text content when native
    # tool_calls were not returned (models that emit XML in plain text).
    content_text = msg.get("content") or ""
    if not tool_calls and content_text:
        tool_calls = _parse_tool_calls(content_text)
        if tool_calls:
            stop_reason = "tool_use"
            content_text = _strip_tool_call_blocks(content_text)

    usage = TokenUsage(
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )
    return ProviderResponse(
        content=content_text or None,
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
    msg_dict: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    if m.images:
        import base64
        from pathlib import Path
        imgs: list[str] = []
        for img_path in m.images:
            try:
                with Path(img_path).open("rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                imgs.append(encoded)
            except Exception as e:
                logger.error(f"Failed to load image for Ollama {img_path}: {e}")
        if imgs:
            msg_dict["images"] = imgs
    return msg_dict


def _to_ollama_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_json_schema(),
        },
    }


def _try_load_tool_json(raw: str) -> dict | None:
    """Try to parse a JSON object, trimming to the last '}' if needed."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        idx = raw.rfind("}")
        if idx != -1:
            try:
                return json.loads(raw[: idx + 1])
            except json.JSONDecodeError:
                pass
    return None


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from <tool_call>…</tool_call> blocks in plain text."""
    calls: list[ToolCall] = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?)\s*</tool_call>", text, re.DOTALL):
        raw = match.group(1).strip()
        data = _try_load_tool_json(raw)
        if data is None:
            continue
        try:
            calls.append(ToolCall(
                id=str(uuid.uuid4()),
                name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except KeyError:
            continue
    return calls


def _strip_tool_call_blocks(text: str) -> str:
    """Remove all <tool_call>…</tool_call> blocks from text."""
    return re.sub(r"<tool_call>.*?</tool_call>\s*", "", text, flags=re.DOTALL).strip()
