"""Tests for Event serialization and non-interactive output modes."""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from tau.core.types import (
    CompactionEvent,
    ErrorEvent,
    ExtensionLoadError,
    RetryEvent,
    SteerEvent,
    TextChunk,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnComplete,
)


# ---------------------------------------------------------------------------
# Event.to_dict() tests
# ---------------------------------------------------------------------------

class TestEventSerialization:
    def test_text_delta(self):
        e = TextDelta(text="hello", is_thinking=False)
        d = e.to_dict()
        assert d == {"type": "text_delta", "text": "hello", "is_thinking": False}

    def test_text_delta_thinking(self):
        e = TextDelta(text="hmm", is_thinking=True)
        d = e.to_dict()
        assert d["is_thinking"] is True

    def test_text_chunk(self):
        e = TextChunk(text="full response")
        d = e.to_dict()
        assert d == {"type": "text_chunk", "text": "full response"}

    def test_tool_call_event(self):
        call = ToolCall(id="tc1", name="read_file", arguments={"path": "/tmp/x"})
        e = ToolCallEvent(call=call)
        d = e.to_dict()
        assert d["type"] == "tool_call"
        assert d["call"]["name"] == "read_file"
        assert d["call"]["arguments"]["path"] == "/tmp/x"

    def test_tool_result_event(self):
        result = ToolResult(tool_call_id="tc1", content="file contents", is_error=False)
        e = ToolResultEvent(result=result)
        d = e.to_dict()
        assert d["type"] == "tool_result"
        assert d["result"]["content"] == "file contents"
        assert d["result"]["is_error"] is False

    def test_turn_complete(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        e = TurnComplete(usage=usage)
        d = e.to_dict()
        assert d["type"] == "turn_complete"
        assert d["usage"]["input_tokens"] == 100
        assert d["usage"]["output_tokens"] == 50

    def test_error_event(self):
        e = ErrorEvent(message="something went wrong")
        d = e.to_dict()
        assert d == {"type": "error", "message": "something went wrong"}

    def test_compaction_event(self):
        e = CompactionEvent(stage="end", tokens_before=1000, tokens_after=500, summary="compressed")
        d = e.to_dict()
        assert d["type"] == "compaction"
        assert d["stage"] == "end"
        assert d["tokens_before"] == 1000
        assert d["tokens_after"] == 500

    def test_retry_event(self):
        e = RetryEvent(attempt=2, max_attempts=3, delay=4.0, error="rate limit")
        d = e.to_dict()
        assert d["type"] == "retry"
        assert d["attempt"] == 2
        assert d["delay"] == 4.0

    def test_steer_event(self):
        e = SteerEvent(new_input="do something else", discarded_tokens=42)
        d = e.to_dict()
        assert d["type"] == "steer"
        assert d["new_input"] == "do something else"

    def test_extension_load_error(self):
        e = ExtensionLoadError(extension_name="bad_ext", error="import failed")
        d = e.to_dict()
        assert d["type"] == "extension_load_error"

    def test_all_events_produce_valid_json(self):
        """Every event should round-trip through JSON without error."""
        events = [
            TextDelta(text="x"),
            TextChunk(text="x"),
            ToolCallEvent(call=ToolCall(id="1", name="t", arguments={})),
            ToolResultEvent(result=ToolResult(tool_call_id="1", content="ok")),
            TurnComplete(usage=TokenUsage()),
            ErrorEvent(message="err"),
            CompactionEvent(stage="start"),
            RetryEvent(attempt=1, max_attempts=3, delay=1.0, error="e"),
            SteerEvent(new_input="x", discarded_tokens=0),
            ExtensionLoadError(extension_name="x", error="y"),
        ]
        for event in events:
            serialized = json.dumps(event.to_dict(), ensure_ascii=False)
            parsed = json.loads(serialized)
            assert "type" in parsed


# ---------------------------------------------------------------------------
# Renderer tests using a fake agent
# ---------------------------------------------------------------------------

def _fake_agent_run(events):
    """Return a mock Agent whose .run() yields the given events."""
    agent = MagicMock()
    agent.run.return_value = iter(events)
    agent._session.cumulative_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
    }
    agent._config.model = "test-model"
    return agent


class TestPrintMode:
    def test_collects_text_deltas(self):
        from tau.cli import _render_events_print

        events = [
            TextDelta(text="Hello"),
            TextDelta(text=" world"),
            TurnComplete(usage=TokenUsage()),
        ]
        agent = _fake_agent_run(events)
        captured = StringIO()
        with patch("sys.stdout", captured):
            _render_events_print(agent, "test prompt")
        assert captured.getvalue().strip() == "Hello world"

    def test_collects_text_chunks(self):
        from tau.cli import _render_events_print

        events = [
            TextChunk(text="Full response here."),
            TurnComplete(usage=TokenUsage()),
        ]
        agent = _fake_agent_run(events)
        captured = StringIO()
        with patch("sys.stdout", captured):
            _render_events_print(agent, "test prompt")
        assert "Full response here." in captured.getvalue()

    def test_errors_go_to_stderr(self):
        from tau.cli import _render_events_print

        events = [ErrorEvent(message="boom")]
        agent = _fake_agent_run(events)
        captured_out = StringIO()
        captured_err = StringIO()
        with patch("sys.stdout", captured_out), patch("sys.stderr", captured_err):
            with pytest.raises(SystemExit):
                _render_events_print(agent, "test prompt")
        assert "boom" in captured_err.getvalue()
        assert captured_out.getvalue() == ""


class TestJsonMode:
    def test_emits_jsonl(self):
        from tau.cli import _render_events_json

        events = [
            TextDelta(text="hi"),
            TurnComplete(usage=TokenUsage(input_tokens=10, output_tokens=5)),
        ]
        agent = _fake_agent_run(events)
        captured = StringIO()
        with patch("sys.stdout", captured):
            _render_events_json(agent, "test prompt")
        lines = captured.getvalue().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "text_delta"
        assert first["text"] == "hi"
        second = json.loads(lines[1])
        assert second["type"] == "turn_complete"
        assert second["usage"]["input_tokens"] == 10

    def test_error_exits_nonzero(self):
        from tau.cli import _render_events_json

        events = [ErrorEvent(message="fail")]
        agent = _fake_agent_run(events)
        captured = StringIO()
        with patch("sys.stdout", captured):
            with pytest.raises(SystemExit) as exc_info:
                _render_events_json(agent, "test prompt")
            assert exc_info.value.code == 1
        line = json.loads(captured.getvalue().strip())
        assert line["type"] == "error"


# ---------------------------------------------------------------------------
# Cancel-event tests for _render_events
# ---------------------------------------------------------------------------

class TestCancelEvent:
    """Pressing Escape sets a cancel_event that _render_events checks."""

    def test_cancel_stops_after_first_event(self):
        """When cancel_event is already set, _render_events breaks after the
        first event and outputs the Cancelled message."""
        import threading
        from tau.cli import _render_events

        events = [
            TextDelta(text="Hello"),
            TextDelta(text=" world"),
            TurnComplete(usage=TokenUsage()),
        ]
        agent = _fake_agent_run(events)
        cancel = threading.Event()
        cancel.set()  # pre-set — simulates Escape pressed before first event

        captured: list[str] = []
        _render_events(
            agent, "test prompt",
            output_fn=captured.append,
            cancel_event=cancel,
        )
        output = "".join(captured)
        assert "Cancelled" in output
        # Should NOT contain the full streamed text
        assert "world" not in output

    def test_no_cancel_renders_fully(self):
        """Without cancel_event set, all events render normally."""
        import threading
        from tau.cli import _render_events

        events = [
            TextDelta(text="abc"),
            TurnComplete(usage=TokenUsage()),
        ]
        agent = _fake_agent_run(events)
        cancel = threading.Event()  # NOT set

        captured: list[str] = []
        _render_events(
            agent, "test prompt",
            output_fn=captured.append,
            cancel_event=cancel,
        )
        output = "".join(captured)
        assert "abc" in output
        assert "Cancelled" not in output

    def test_cancel_mid_stream(self):
        """Simulates cancel being set after the first delta."""
        import threading
        from tau.cli import _render_events

        cancel = threading.Event()

        def _events():
            yield TextDelta(text="first")
            cancel.set()  # cancel after first delta
            yield TextDelta(text="second")
            yield TurnComplete(usage=TokenUsage())

        agent = MagicMock()
        agent.run.return_value = _events()

        captured: list[str] = []
        _render_events(
            agent, "test prompt",
            output_fn=captured.append,
            cancel_event=cancel,
        )
        output = "".join(captured)
        assert "first" in output
        assert "Cancelled" in output
        # "second" may or may not appear depending on timing, but
        # TurnComplete should NOT
        assert "tokens:" not in output

    def test_cancel_none_is_safe(self):
        """cancel_event=None (default) works without errors."""
        from tau.cli import _render_events

        events = [
            TextDelta(text="ok"),
            TurnComplete(usage=TokenUsage()),
        ]
        agent = _fake_agent_run(events)

        captured: list[str] = []
        _render_events(
            agent, "test prompt",
            output_fn=captured.append,
            cancel_event=None,
        )
        output = "".join(captured)
        assert "ok" in output
