"""Google Gemini provider."""

from __future__ import annotations

import logging
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
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "error",
    "RECITATION": "error",
    "OTHER": "error",
}


class GoogleProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as exc:
            raise ImportError("google-genai package is required: pip install google-genai") from exc

        self._model = agent_config.model
        self._client = genai.Client(api_key=config.google.api_key or None)
        self._gtypes = gtypes

    @property
    def name(self) -> str:
        return "google"

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse:
        gtypes = self._gtypes
        system_prompt, history, last_user = _split_messages(messages)

        google_tools = _to_google_tools(tools, gtypes) if tools else None
        config_kwargs: dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if google_tools:
            config_kwargs["tools"] = google_tools
        gen_config = gtypes.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        contents = history + [{"role": "user", "parts": [{"text": last_user}]}]
        logger.debug("Google request: model=%s, turns=%d, stream=%s",
                     self._model, len(history), stream)

        if stream:
            return self._chat_stream(contents, gen_config)
        return self._chat_blocking(contents, gen_config)

    # ------------------------------------------------------------------

    def _chat_blocking(self, contents, gen_config) -> ProviderResponse:
        response = self._client.models.generate_content(
            model=self._model, contents=contents, config=gen_config,
        )
        return _parse_google_response(response)

    def _chat_stream(self, contents, gen_config) -> ProviderResponse:
        content_parts: list[str] = []
        last_response = None

        for chunk in self._client.models.generate_content_stream(
            model=self._model, contents=contents, config=gen_config,
        ):
            last_response = chunk
            # Each chunk may have text parts
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append(part.text)
                        yield TextDelta(text=part.text)  # type: ignore[misc]

        # Parse final accumulated response for tool calls, usage, stop reason
        if last_response is not None:
            result = _parse_google_response(last_response)
            # Override content with the full accumulated stream text
            yield ProviderResponse(  # type: ignore[misc]
                content="".join(content_parts) or result.content,
                tool_calls=result.tool_calls,
                stop_reason=result.stop_reason,
                usage=result.usage,
            )
        else:
            yield ProviderResponse(  # type: ignore[misc]
                content="".join(content_parts) or None,
                tool_calls=[],
                stop_reason="end_turn",
                usage=TokenUsage(),
            )


# ---------------------------------------------------------------------------
# Shared response parser
# ---------------------------------------------------------------------------

def _parse_google_response(response) -> ProviderResponse:
    candidate = response.candidates[0]
    finish = getattr(candidate, "finish_reason", "STOP")
    stop_reason: StopReason = _STOP_REASON_MAP.get(str(finish), "end_turn")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        elif hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(ToolCall(
                id=fc.id if hasattr(fc, "id") else fc.name,
                name=fc.name,
                arguments=dict(fc.args) if fc.args else {},
            ))

    if tool_calls:
        stop_reason = "tool_use"

    usage_meta = getattr(response, "usage_metadata", None)
    usage = TokenUsage(
        input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
    )
    return ProviderResponse(
        content="\n".join(text_parts) or None,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_messages(messages: list[Message]) -> tuple[str, list[dict], str]:
    system_parts: list[str] = []
    history: list[dict] = []
    last_user = ""

    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            last_user = m.content
            if history:
                history.append({"role": "user", "parts": [{"text": m.content}]})
        elif m.role == "assistant":
            history.append({"role": "model", "parts": [{"text": m.content or ""}]})
        elif m.role == "tool":
            history.append({
                "role": "user",
                "parts": [{"function_response": {
                    "id": m.tool_call_id,
                    "name": m.name or "tool",
                    "response": {"output": m.content},
                }}],
            })

    return "\n\n".join(system_parts), history, last_user


def _to_google_tools(tools: list[ToolDefinition], gtypes: Any) -> list[Any]:
    declarations = []
    for t in tools:
        declarations.append(gtypes.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.to_json_schema(),
        ))
    return [gtypes.Tool(function_declarations=declarations)]
