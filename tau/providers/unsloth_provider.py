"""Unsloth Studio provider — uses llama-server's OpenAI-compatible API.

Unsloth Studio serves models locally via llama-server (llama.cpp) which
exposes an OpenAI-compatible /v1/chat/completions endpoint.

Launch Unsloth Studio:
    unsloth studio -H 0.0.0.0 -p 8910

Or run llama-server directly:
    llama-server --model your_model.gguf --port 8001

Configure in ~/.tau/config.toml:
    [defaults]
    provider = "unsloth"
    model    = "local-model"

    [providers.unsloth]
    base_url = "http://localhost:8001/v1"     # llama-server default
"""

from __future__ import annotations

import json
import logging
import uuid
import re
import time
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
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "error",
}


def _estimate_tokens(text: str) -> int:
    # Fast fallback estimator for local providers that omit token usage.
    return max(1, len((text or "").strip()) // 4) if (text or "").strip() else 0


def _message_text_content(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join([p for p in parts if p])
    return str(content or "")


def _estimate_prompt_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_tokens(_message_text_content(m)) for m in messages if isinstance(m, dict))


_THINK_START_TAGS = ("<think>", "<|think|>")
_THINK_END_TAG = "</think>"


def _split_stream_text_for_thinking(
    text: str,
    *,
    in_thinking: bool,
) -> tuple[str, str, bool, str]:
    """Split streamed text into visible/thinking fragments with carry support.

    Returns:
      (visible_text, thinking_text, new_in_thinking, carry_tail)
    """
    visible_parts: list[str] = []
    think_parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if in_thinking:
            end_idx = text.find(_THINK_END_TAG, i)
            if end_idx == -1:
                remain = text[i:]
                carry = ""
                if _THINK_END_TAG.startswith(remain):
                    carry = remain
                    remain = ""
                if remain:
                    think_parts.append(remain)
                return "".join(visible_parts), "".join(think_parts), True, carry
            think_parts.append(text[i:end_idx])
            i = end_idx + len(_THINK_END_TAG)
            in_thinking = False
            continue

        starts = [(text.find(tag, i), tag) for tag in _THINK_START_TAGS]
        starts = [(idx, tag) for idx, tag in starts if idx != -1]
        if not starts:
            remain = text[i:]
            carry = ""
            for tag in _THINK_START_TAGS:
                max_len = min(len(tag) - 1, len(remain))
                for k in range(max_len, 0, -1):
                    if tag.startswith(remain[-k:]):
                        carry = remain[-k:]
                        remain = remain[:-k]
                        break
                if carry:
                    break
            if remain:
                visible_parts.append(remain)
            return "".join(visible_parts), "".join(think_parts), False, carry

        next_idx, next_tag = min(starts, key=lambda x: x[0])
        if next_idx > i:
            visible_parts.append(text[i:next_idx])
        i = next_idx + len(next_tag)
        in_thinking = True

    return "".join(visible_parts), "".join(think_parts), in_thinking, ""


class UnslothProvider:
    """Provider for Unsloth Studio / llama-server (OpenAI-compatible)."""

    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        self._model = agent_config.model
        self._base_url = config.unsloth.base_url.rstrip("/")
        timeout_s = max(5.0, float(config.unsloth.timeout_seconds))
        stream_read_timeout_s = float(config.unsloth.stream_read_timeout_seconds)

        # Default request timeout for non-streaming calls.
        self._request_timeout = httpx.Timeout(timeout_s)

        # Streaming should tolerate long prefill before first token.
        # Keep connect/write/pool bounded while allowing read timeout
        # to be disabled (default) for local llama-server workloads.
        stream_read_timeout = None if stream_read_timeout_s <= 0 else max(1.0, stream_read_timeout_s)
        self._stream_timeout = httpx.Timeout(
            connect=timeout_s,
            read=stream_read_timeout,
            write=timeout_s,
            pool=timeout_s,
        )

        self._client = httpx.Client(timeout=self._request_timeout)
        self._stream_yield_every_chunks = max(0, int(config.unsloth.stream_yield_every_chunks))
        self._stream_yield_s = max(0.0, float(config.unsloth.stream_yield_ms) / 1000.0)
        self._last_response_headers: dict[str, str] = {}

    @property
    def last_response_headers(self) -> dict[str, str]:
        return self._last_response_headers

    import contextlib
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
        return "unsloth"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse | Generator:
        oai_messages = [_to_oai_message(m) for m in messages]
        oai_tools = [_to_oai_tool(t) for t in tools] if tools else []

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "stream": stream,
        }
        if oai_tools:
            payload["tools"] = oai_tools

        logger.debug(
            "Unsloth request: base_url=%s, model=%s, messages=%d, stream=%s",
            self._base_url, self._model, len(oai_messages), stream,
        )

        if stream:
            return self._chat_stream(payload)
        return self._chat_blocking(payload)

    # ------------------------------------------------------------------

    def _chat_blocking(self, payload: dict[str, Any]) -> ProviderResponse:
        resp = self._client.post(
            f"{self._base_url}/chat/completions", json=payload,
        )
        self._last_response_headers = dict(resp.headers)
        resp.raise_for_status()
        data = resp.json()
        parsed = _parse_response(data)
        if parsed.usage.input_tokens == 0 and parsed.usage.output_tokens == 0:
            parsed.usage = TokenUsage(
                input_tokens=_estimate_prompt_tokens_from_messages(payload.get("messages", [])),
                output_tokens=_estimate_tokens(parsed.content or ""),
            )
        return parsed

    def _chat_stream(self, payload: dict[str, Any]) -> Generator:
        payload["stream"] = True

        content_parts: list[str] = []
        tc_acc: dict[int, dict[str, Any]] = {}
        stop_reason: StopReason = "end_turn"
        usage = TokenUsage()
        yielded_chunks = 0
        in_thinking = False
        think_carry = ""

        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=self._stream_timeout,
            ) as resp:
                self._last_response_headers = dict(resp.headers)
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Usage (if provided)
                    if "usage" in chunk and chunk["usage"]:
                        u = chunk["usage"]
                        usage = TokenUsage(
                            input_tokens=u.get("prompt_tokens", 0),
                            output_tokens=u.get("completion_tokens", 0),
                        )

                    if not chunk.get("choices"):
                        continue

                    choice = chunk["choices"][0]

                    if choice.get("finish_reason"):
                        stop_reason = _STOP_REASON_MAP.get(
                            choice["finish_reason"], "end_turn",
                        )

                    delta = choice.get("delta", {})

                    # Native reasoning stream fields from OpenAI-compatible servers.
                    reasoning = delta.get("reasoning_content") or delta.get("thinking") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        yield TextDelta(text=reasoning, is_thinking=True)  # type: ignore[misc]

                    # Text delta
                    text = delta.get("content")
                    if text:
                        parsed_text = think_carry + text
                        think_carry = ""
                        visible, think, in_thinking, think_carry = _split_stream_text_for_thinking(
                            parsed_text,
                            in_thinking=in_thinking,
                        )
                        if think:
                            yield TextDelta(text=think, is_thinking=True)  # type: ignore[misc]
                        if visible:
                            content_parts.append(visible)
                            yielded_chunks += 1
                            yield TextDelta(text=visible)  # type: ignore[misc]
                        if (
                            self._stream_yield_every_chunks > 0
                            and self._stream_yield_s > 0.0
                            and yielded_chunks % self._stream_yield_every_chunks == 0
                        ):
                            # Cooperative pause to keep local desktop/UI responsive
                            # when Unsloth runs on the same machine.
                            time.sleep(self._stream_yield_s)

                    # Tool call deltas
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tc_acc:
                                tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.get("id"):
                                tc_acc[idx]["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                tc_acc[idx]["name"] += fn["name"]
                            if fn.get("arguments"):
                                tc_acc[idx]["arguments"] += fn["arguments"]
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                "Unsloth stream read timed out before completion. "
                "For long prefill prompts, set UNSLOTH_STREAM_READ_TIMEOUT_SECONDS=0 "
                "(disable read timeout) or increase it in config."
            ) from exc

        tool_calls = [
            ToolCall(
                id=v["id"] or str(uuid.uuid4()),
                name=v["name"],
                arguments=json.loads(v["arguments"] or "{}"),
            )
            for v in tc_acc.values()
        ]

        # Fallback: parse <tool_call> blocks from text when native tool_calls
        # are not returned (some local models emit XML in plain text).
        if think_carry and not in_thinking:
            content_parts.append(think_carry)
        full_text = "".join(content_parts)
        if not tool_calls and full_text:
            tool_calls = _parse_tool_calls(full_text)
            if tool_calls:
                stop_reason = "tool_use"
                full_text = _strip_tool_call_blocks(full_text)

        if tool_calls:
            stop_reason = "tool_use"

        if usage.input_tokens == 0 and usage.output_tokens == 0:
            usage = TokenUsage(
                input_tokens=_estimate_prompt_tokens_from_messages(payload.get("messages", [])),
                output_tokens=_estimate_tokens(full_text),
            )

        yield ProviderResponse(  # type: ignore[misc]
            content=full_text or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Response parser (non-streaming)
# ---------------------------------------------------------------------------

def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    choice = data["choices"][0]
    msg = choice["message"]
    finish = choice.get("finish_reason", "stop")
    stop_reason: StopReason = _STOP_REASON_MAP.get(finish, "end_turn")

    tool_calls: list[ToolCall] = []
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(
                id=tc.get("id", str(uuid.uuid4())),
                name=fn.get("name", ""),
                arguments=args,
            ))
        stop_reason = "tool_use"

    content_text = msg.get("content") or ""

    # Fallback: parse <tool_call> blocks from text content
    if not tool_calls and content_text:
        tool_calls = _parse_tool_calls(content_text)
        if tool_calls:
            stop_reason = "tool_use"
            content_text = _strip_tool_call_blocks(content_text)

    u = data.get("usage", {})
    usage = TokenUsage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
    )
    return ProviderResponse(
        content=content_text or None,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Message / tool conversion helpers
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
    content: str | list[dict[str, Any]] = m.content or ""
    if m.images:
        import base64
        import mimetypes
        from pathlib import Path
        parts: list[dict[str, Any]] = [{"type": "text", "text": content}] if content else []
        for img_path in m.images:
            try:
                mime_type, _ = mimetypes.guess_type(img_path)
                mime_type = mime_type or "image/jpeg"
                with Path(img_path).open("rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                })
            except Exception as e:
                logger.error("Failed to load image for Unsloth %s: %s", img_path, e)
        if parts:
            content = parts
    return {"role": m.role, "content": content}


def _to_oai_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_json_schema(),
        },
    }


# ---------------------------------------------------------------------------
# <tool_call> XML fallback parsing (for models without native tool support)
# ---------------------------------------------------------------------------

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
    for match in re.finditer(r"<tool_call>\s*(\{.*?})\s*</tool_call>", text, re.DOTALL):
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
