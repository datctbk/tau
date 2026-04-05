"""MLX provider — runs models locally on Apple Silicon via mlx-lm."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import Generator
from typing import Any

# Suppress the harmless macOS malloc stack logging noise emitted by Metal/MLX.
os.environ.setdefault("MallocStackLogging", "0")

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


def _is_vlm_model(model_name: str) -> bool:
    """Return True for models that require mlx-vlm (e.g. Gemma 4)."""
    lower = model_name.lower()
    return "gemma4" in lower or "gemma-4" in lower


class MLXProvider:
    def __init__(self, config: TauConfig, agent_config: AgentConfig) -> None:
        self._model_name = agent_config.model
        self._use_vlm = _is_vlm_model(self._model_name)

        if self._use_vlm:
            try:
                import mlx_vlm  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "mlx-vlm package is required for this model: pip install mlx-vlm"
                ) from exc
        else:
            try:
                import mlx_lm  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "mlx-lm package is required: pip install mlx-lm"
                ) from exc

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

        logger.info("MLX: loading model %s …", self._model_name)
        import io, os, sys
        # Suppress all tqdm / huggingface_hub progress bars that leave
        # ghost lines on the interactive terminal.
        _env_save = {}
        for key in ("HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE"):
            _env_save[key] = os.environ.get(key)
            os.environ[key] = "1"
        # Redirect stderr to swallow any residual progress output
        _real_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            if self._use_vlm:
                from mlx_vlm import load
                model, processor = load(self._model_name)
                self._ensure_chat_template(processor, model)
                _cached[self._model_name] = (model, processor)
                return model, processor
            else:
                from mlx_lm import load
                model, tokenizer = load(self._model_name)
                _cached[self._model_name] = (model, tokenizer)
                return model, tokenizer
        finally:
            sys.stderr = _real_stderr
            for key, old_val in _env_save.items():
                if old_val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_val
            # Clear any leftover line on stdout
            try:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            except Exception:
                pass

    def _get_tokenizer(self) -> Any:
        """Return the actual tokenizer for apply_chat_template / encode.

        For mlx-vlm models the loader returns a processor whose underlying
        HuggingFace tokenizer lives at processor.tokenizer.
        """
        if self._use_vlm:
            return self._tokenizer.tokenizer
        return self._tokenizer

    @staticmethod
    def _ensure_chat_template(processor: Any, model: Any) -> None:
        """Ensure the VLM processor has a chat template.

        The mlx-community quantized Gemma 4 models ship without a Jinja
        chat template.  If one is missing we try to download the official
        template from the Google conversion reference repo.
        """
        if getattr(processor, "chat_template", None) is not None:
            return

        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if model_type != "gemma4":
            return

        _TEMPLATE_REPOS = [
            "gg-hf-gg/gemma-4-E4B-it",
            "gg-hf-gg/gemma-4-31B-it",
        ]
        for repo in _TEMPLATE_REPOS:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo, "chat_template.jinja")
                with open(path) as f:
                    processor.chat_template = f.read()
                logger.info("MLX: loaded Gemma 4 chat template from %s", repo)
                return
            except Exception:
                continue

        logger.warning(
            "MLX: could not download Gemma 4 chat template; "
            "falling back to manual turn format"
        )

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
        chat_messages = [_to_chat_message(m, gemma4=self._use_vlm) for m in messages]

        if self._use_vlm:
            return self._build_vlm_prompt(chat_messages, tools)

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._enable_thinking:
            kwargs["enable_thinking"] = True
        if tools:
            kwargs["tools"] = [_to_tool_dict(t) for t in tools]

        return self._get_tokenizer().apply_chat_template(chat_messages, **kwargs)

    def _build_vlm_prompt(self, chat_messages: list[dict[str, Any]], tools: list[ToolDefinition]) -> str:
        """Build prompt for VLM models (e.g. Gemma 4).

        Tries, in order:
        1. processor.apply_chat_template (Jinja template set during loading)
        2. processor.tokenizer.apply_chat_template
        3. Manual Gemma 4 turn format as last resort
        """
        processor = self._tokenizer  # the AutoProcessor from mlx_vlm.load

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = [_to_tool_dict(t) for t in tools]
        if self._enable_thinking:
            kwargs["enable_thinking"] = True

        # Strategy 1: processor itself (has chat_template after _ensure_chat_template)
        if getattr(processor, "chat_template", None) is not None:
            try:
                return processor.apply_chat_template(
                    chat_messages, **kwargs,
                )
            except Exception:
                pass

        # Strategy 2: underlying tokenizer
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "chat_template", None) is not None:
            try:
                return tokenizer.apply_chat_template(
                    chat_messages, **kwargs,
                )
            except Exception:
                pass

        # Strategy 3: manual Gemma 4 turn format
        logger.warning(
            "MLX VLM: no chat_template found on processor; "
            "falling back to manual Gemma 4 turn format."
        )
        return _build_gemma_prompt(chat_messages)

    def _vlm_eos_tokens(self) -> list[str]:
        """Return extra EOS tokens for VLM models.

        Gemma 4 uses ``<turn|>`` (eot_token) to end a turn, but it is not
        included in the default eos_token_id.  Adding it here makes the
        model stop generating once its turn is complete.
        """
        tokenizer = self._get_tokenizer()
        eot = getattr(tokenizer, "eot_token", None)
        return [eot] if eot else []

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
        if self._use_vlm:
            from mlx_vlm import generate
            result = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                eos_tokens=self._vlm_eos_tokens(),
            )
            raw = result.text
            prompt_tokens = result.prompt_tokens
            output_tokens = result.generation_tokens
        else:
            from mlx_lm import generate
            raw = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                sampler=self._make_sampler(),
                logits_processors=self._make_logits_processors(),
            )
            tokenizer = self._get_tokenizer()
            prompt_tokens = len(tokenizer.encode(prompt))
            output_tokens = len(tokenizer.encode(raw))

        content, _thinking = _split_thinking(raw)
        tool_calls = _parse_tool_calls(content)
        stop_reason: StopReason = "tool_use" if tool_calls else "end_turn"
        if tool_calls:
            content = _strip_tool_call_blocks(content)

        return ProviderResponse(
            content=content or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens),
        )

    def _chat_stream(self, prompt: str) -> Generator:
        content_parts: list[str] = []
        buffer = ""
        # Qwen3 models always emit <think>...</think> blocks.
        # If the prompt already ends with <think> (the chat template appended
        # it), we are already inside the block — no opening tag will appear.
        in_thinking = prompt.rstrip().endswith("<think>") or prompt.rstrip().endswith("<|think|>")
        think_search_done = in_thinking  # already committed to think state
        in_tool_call = False              # suppress tool-call markup from display
        # Gemma 4 uses <|tool_call>…<tool_call|>, Qwen uses <tool_call>…</tool_call>
        _TC_OPEN = "<|tool_call>" if self._use_vlm else "<tool_call>"
        _TC_CLOSE = "<tool_call|>" if self._use_vlm else "</tool_call>"
        tokenizer = self._get_tokenizer()
        prompt_tokens = len(tokenizer.encode(prompt))
        output_tokens = 0
        last_vlm_response: Any = None

        if self._use_vlm:
            from mlx_vlm import stream_generate
            stream_iter = stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                eos_tokens=self._vlm_eos_tokens(),
            )
        else:
            from mlx_lm import stream_generate
            stream_iter = stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_tokens,
                sampler=self._make_sampler(),
                logits_processors=self._make_logits_processors(),
            )

        for response in stream_iter:
            token = response.text if hasattr(response, "text") else str(response)
            if self._use_vlm:
                last_vlm_response = response
            output_tokens += 1
            buffer += token

            # --- State: scanning for opening <think> ----------------------
            if not think_search_done and not in_thinking:
                if "<think>" in buffer:
                    think_search_done = True
                    in_thinking = True
                    before, _, after = buffer.partition("<think>")
                    if before.strip():
                        content_parts.append(before)
                        yield TextDelta(text=before)
                    buffer = after
                    if buffer:
                        yield TextDelta(text=buffer, is_thinking=True)
                        buffer = ""
                    continue
                # Buffer enough chars; if no <think> found, treat as content
                if len(buffer) > 12:
                    think_search_done = True
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
                    think_search_done = True
                    buffer = rest.lstrip("\n")
                    if buffer:
                        content_parts.append(buffer)
                        yield TextDelta(text=buffer)
                        buffer = ""
                else:
                    yield TextDelta(text=token, is_thinking=True)
                    buffer = ""
                continue

            # --- State: inside tool_call block (suppress from display) --
            if in_tool_call:
                if _TC_CLOSE in buffer:
                    in_tool_call = False
                    # Keep accumulated buffer for final parsing; emit nothing
                    content_parts.append(buffer)
                    buffer = ""
                # else: still accumulating — don't emit anything
                continue

            # --- State: normal content — watch for tool_call opening ----
            if _TC_OPEN in buffer:
                before, _, after = buffer.partition(_TC_OPEN)
                if before:
                    content_parts.append(before)
                    yield TextDelta(text=before)
                # Re-attach the tag so _parse_tool_calls can find it later
                buffer = _TC_OPEN + after
                in_tool_call = True
                continue

            # Hold partial tag prefixes in the buffer rather than emitting early
            if "<" in buffer and not buffer.endswith(">") and not buffer.endswith("\n"):
                continue

            content_parts.append(token)
            yield TextDelta(text=token)
            buffer = ""

        # Flush remaining buffer
        if buffer:
            stripped = buffer.replace("</think>", "")
            if in_thinking:
                if stripped:
                    yield TextDelta(text=stripped, is_thinking=True)
            elif not in_tool_call:
                if stripped:
                    content_parts.append(stripped)
                    yield TextDelta(text=stripped)
            else:
                # Incomplete tool_call block — still keep for parsing
                content_parts.append(buffer)

        final_content = "".join(content_parts).strip()
        tool_calls = _parse_tool_calls(final_content)
        stop_reason: StopReason = "tool_use" if tool_calls else "end_turn"
        if tool_calls:
            final_content = _strip_tool_call_blocks(final_content)

        if self._use_vlm and last_vlm_response is not None:
            prompt_tokens = last_vlm_response.prompt_tokens
            output_tokens = last_vlm_response.generation_tokens

        yield ProviderResponse(
            content=final_content or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Gemma 4 uses <|turn>role\n…<turn|> markers (different from Gemma 3).
_GEMMA_ROLE_MAP = {"system": "system", "user": "user", "assistant": "model", "tool": "tool"}

def _build_gemma_prompt(messages: list[dict[str, Any]]) -> str:
    """Build a Gemma 4 prompt from chat messages when no chat_template is available."""
    parts: list[str] = []
    for m in messages:
        role = _GEMMA_ROLE_MAP.get(m.get("role", "user"), m.get("role", "user"))
        content = m.get("content", "") or ""
        parts.append(f"<|turn>{role}\n{content}<turn|>")
    # Add generation prompt for the model to start responding
    parts.append("<|turn>model\n")
    return "\n".join(parts)


def _to_chat_message(m: Message, *, gemma4: bool = False) -> dict[str, Any]:
    if m.role == "tool":
        if gemma4:
            # Gemma 4 Jinja template expects tool results as tool_responses
            # on the preceding assistant message.  However, the tau agent adds
            # them as separate "tool" role messages.  The Jinja template
            # accepts a list of dicts under the "tool" role.
            return {
                "role": "tool",
                "content": m.content,
                "name": m.name or "",
            }
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
    """Split response into (content, thinking).

    Supports both Qwen-style ``<think>…</think>`` and Gemma 4 ``<|think|>…``
    (Gemma 4 uses the ``<|think|>`` token to start thinking; generation stops
    at ``<turn|>`` so there may be no explicit close marker).
    """
    # Qwen: full <think>…</think> block present
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        content = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        return content, thinking
    # Qwen: only </think> (opening tag was in the prompt)
    if "</think>" in text:
        thinking_text, _, content = text.partition("</think>")
        return content.strip(), thinking_text.strip() or None
    return text.strip(), None


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """Best-effort extraction of tool calls from model output.

    Supports three formats:
      1. <tool_call>{"name": …, "arguments": …}</tool_call>  (Qwen/generic)
      2. <|tool_call>call:name{arg:<|"|>val<|"|>}<tool_call|>  (Gemma 4)
      3. Qwen-style  ✿FUNCTION✿: name / ✿ARGS✿: {…}
    """
    calls: list[ToolCall] = []

    # Pattern 1: <tool_call> blocks (Qwen / generic JSON)
    for match in re.finditer(
        r"<tool_call>\s*(\{.*?)\s*</tool_call>", text, re.DOTALL
    ):
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

    if calls:
        return calls

    # Pattern 2: Gemma 4 — <|tool_call>call:name{key:<|"|>val<|"|>,...}<tool_call|>
    for match in re.finditer(
        r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", text, re.DOTALL
    ):
        name = match.group(1)
        raw_args = match.group(2)
        args = _parse_gemma4_args(raw_args)
        calls.append(ToolCall(
            id=str(uuid.uuid4()),
            name=name,
            arguments=args,
        ))

    if calls:
        return calls

    # Pattern 3: Qwen function calling format
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


def _parse_gemma4_args(raw: str) -> dict[str, Any]:
    """Parse Gemma 4 key-value tool-call arguments.

    Gemma 4 uses a custom syntax:
        key1:<|"|>string value<|"|>,key2:42,key3:true
    Values wrapped in <|"|>…<|"|> are strings; bare values are cast to
    int/float/bool where possible.
    """
    args: dict[str, Any] = {}
    # Tokenise: alternate between <|"|>…<|"|> quoted strings and bare segments
    i = 0
    key: str | None = None
    while i < len(raw):
        # Skip leading commas/whitespace
        if raw[i] in (",", " ", "\n"):
            i += 1
            continue
        # Find key (word before colon)
        m = re.match(r"(\w+):", raw[i:])
        if m:
            key = m.group(1)
            i += m.end()
            # Check for quoted string value
            if raw[i:].startswith('<|"|>'):
                end_q = raw.find('<|"|>', i + 5)
                if end_q != -1:
                    args[key] = raw[i + 5 : end_q]
                    i = end_q + 5
                else:
                    args[key] = raw[i + 5 :]
                    break
            else:
                # Bare value — up to next comma or end
                end_v = raw.find(",", i)
                if end_v == -1:
                    end_v = len(raw)
                val_str = raw[i:end_v].strip()
                args[key] = _cast_bare_value(val_str)
                i = end_v + 1
        else:
            i += 1
    return args


def _cast_bare_value(v: str) -> Any:
    """Cast a bare string value to int, float, or bool if possible."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _try_load_tool_json(raw: str) -> dict | None:
    """Try to parse a JSON tool-call object, repairing common model output issues.

    Handles:
    - Unescaped literal newlines/tabs inside JSON string values
    - Truncated output (missing one or more closing braces)
    """
    # Attempt 1: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: escape literal control characters that models sometimes emit
    # unescaped inside string values (e.g. actual newline chars instead of \n)
    fixed = raw.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: model may have been truncated — try appending closing braces
    for extra in ("}", "}}", "}}}"):
        try:
            return json.loads(fixed + extra)
        except json.JSONDecodeError:
            continue

    return None


def _strip_tool_call_blocks(text: str) -> str:
    """Remove tool-call markup from visible content."""
    # Qwen / generic
    text = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"✿FUNCTION✿:.*?(?=✿FUNCTION✿:|$)", "", text, flags=re.DOTALL)
    # Gemma 4
    text = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", text, flags=re.DOTALL)
    # Also strip the <|tool_response> tag the model may emit right after a call
    text = re.sub(r"<\|tool_response>", "", text)
    return text.strip()
