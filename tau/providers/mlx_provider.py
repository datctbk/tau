"""MLX provider — runs models locally on Apple Silicon via mlx-lm."""

from __future__ import annotations

import json
import logging
import re
import uuid
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

# Module-level cache — model loading is expensive; reuse across turns.
_cached: dict[str, tuple[Any, Any]] = {}


class MLXProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        try:
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mlx-lm package is required: pip install mlx-lm"
            ) from exc

        self._model_name = agent_config.model
        self._max_tokens = agent_config.max_tokens
        self._enable_thinking = agent_config.thinking_level != "off"
        self._repetition_penalty = config.mlx.repetition_penalty
        self._top_p = config.mlx.top_p
        self._temperature = config.mlx.temperature
        self._model, self._tokenizer = self._load_model()

    def _load_model(self) -> tuple[Any, Any]:
        if self._model_name in _cached:
            logger.info("MLX: reusing cached model %s", self._model_name)
            return _cached[self._model_name]

        from mlx_lm import load

        logger.info("MLX: loading model %s …", self._model_name)
        model, tokenizer = load(self._model_name)
        _cached[self._model_name] = (model, tokenizer)
        return model, tokenizer

    # ---- Protocol --------------------------------------------------------

    @property
    def name(self) -> str:
        return "mlx"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse | Generator:
        prompt = self._build_prompt(messages, tools)
        logger.debug(
            "MLX request: model=%s, prompt_len=%d, stream=%s",
            self._model_name, len(prompt), stream,
        )
        if stream:
            return self._chat_stream(prompt)
        return self._chat_blocking(prompt)

    # ---- Internals -------------------------------------------------------

    def _build_prompt(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> str:
        chat_messages = [_to_chat_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        if tools:
            kwargs["tools"] = [_to_tool_dict(t) for t in tools]

        return self._tokenizer.apply_chat_template(chat_messages, **kwargs)

    def _make_sampler(self):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=self._temperature, top_p=self._top_p)

    def _make_logits_processors(self):
        from mlx_lm.sample_utils import make_logits_processors
        processors = make_logits_processors(
            repetition_penalty=self._repetition_penalty,
        )
        return processors if processors else None

    def _chat_blocking(self, prompt: str) -> ProviderResponse:
        from mlx_lm import generate

        raw = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
            sampler=self._make_sampler(),
            logits_processors=self._make_logits_processors(),
        )

        content, _thinking = _split_thinking(raw)
        tool_calls = _parse_tool_calls(content)
        stop_reason: StopReason = "tool_use" if tool_calls else "end_turn"
        if tool_calls:
            content = _strip_tool_call_blocks(content)

        prompt_tokens = len(self._tokenizer.encode(prompt))
        output_tokens = len(self._tokenizer.encode(raw))

        return ProviderResponse(
            content=content or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens),
        )

    def _chat_stream(self, prompt: str) -> Generator:
        from mlx_lm import stream_generate

        content_parts: list[str] = []
        buffer = ""
        in_thinking = False
        thinking_done = False
        prompt_tokens = len(self._tokenizer.encode(prompt))
        output_tokens = 0

        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
            sampler=self._make_sampler(),
            logits_processors=self._make_logits_processors(),
        ):
            token = response.text if hasattr(response, "text") else str(response)
            output_tokens += 1
            buffer += token

            # --- State: waiting to see if a <think> block starts ----------
            if not thinking_done and not in_thinking:
                if "<think>" in buffer:
                    in_thinking = True
                    # Anything before <think> is content
                    before, _, after = buffer.partition("<think>")
                    if before.strip():
                        content_parts.append(before)
                        yield TextDelta(text=before)
                    buffer = after
                    if buffer:
                        yield TextDelta(text=buffer, is_thinking=True)
                        buffer = ""
                    continue
                # If we've buffered enough without seeing <think>, give up waiting
                if len(buffer) > 12:
                    thinking_done = True
                    content_parts.append(buffer)
                    yield TextDelta(text=buffer)
                    buffer = ""
                continue

            # --- State: inside <think> block ------------------------------
            if in_thinking:
                if "</think>" in buffer:
                    thinking_text, _, rest = buffer.partition("</think>")
                    if thinking_text:
                        yield TextDelta(text=thinking_text, is_thinking=True)
                    in_thinking = False
                    thinking_done = True
                    buffer = rest.lstrip("\n")
                    if buffer:
                        content_parts.append(buffer)
                        yield TextDelta(text=buffer)
                        buffer = ""
                else:
                    yield TextDelta(text=token, is_thinking=True)
                    buffer = ""
                continue

            # --- State: normal content ------------------------------------
            content_parts.append(token)
            yield TextDelta(text=token)
            buffer = ""

        # Flush remaining buffer
        if buffer:
            if in_thinking:
                yield TextDelta(text=buffer, is_thinking=True)
            else:
                content_parts.append(buffer)
                yield TextDelta(text=buffer)

        final_content = "".join(content_parts).strip()
        tool_calls = _parse_tool_calls(final_content)
        stop_reason: StopReason = "tool_use" if tool_calls else "end_turn"
        if tool_calls:
            final_content = _strip_tool_call_blocks(final_content)

        yield ProviderResponse(
            content=final_content or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_chat_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "content": m.content, "name": m.name or ""}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content or ""}


def _to_tool_dict(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.to_json_schema(),
        },
    }


def _split_thinking(text: str) -> tuple[str, str | None]:
    """Split response into (content, thinking). Strips <think>…</think>."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        content = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return content, thinking
    return text.strip(), None


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """Best-effort extraction of tool calls from model output.

    Supports two common formats:
      1. <tool_call>{"name": …, "arguments": …}</tool_call>
      2. Qwen-style  ✿FUNCTION✿: name / ✿ARGS✿: {…}
    """
    calls: list[ToolCall] = []

    # Pattern 1: <tool_call> blocks
    for match in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL
    ):
        try:
            data = json.loads(match.group(1))
            calls.append(ToolCall(
                id=str(uuid.uuid4()),
                name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    if calls:
        return calls

    # Pattern 2: Qwen function calling format
    for match in re.finditer(
        r"✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*(\{.*?\})", text, re.DOTALL
    ):
        try:
            args = json.loads(match.group(2))
            calls.append(ToolCall(
                id=str(uuid.uuid4()),
                name=match.group(1),
                arguments=args,
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


def _strip_tool_call_blocks(text: str) -> str:
    """Remove tool-call markup from visible content."""
    text = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"✿FUNCTION✿:.*?(?=✿FUNCTION✿:|$)", "", text, flags=re.DOTALL)
    return text.strip()
