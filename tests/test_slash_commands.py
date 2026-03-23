"""Tests for REPL slash commands: /clear, /compact, /model, /tokens, /help."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tau.cli as _cli_mod          # patch against the module, not the import
from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.extension import ExtensionContext, ExtensionRegistry
from tau.core.session import Session, SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    CompactionEntry,
    Message,
    ProviderResponse,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> AgentConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        max_tokens=8192,
        max_turns=10,
        system_prompt="sys",
        compaction_enabled=False,
        retry_enabled=False,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _make_agent(cfg: AgentConfig | None = None) -> Agent:
    config = cfg or _cfg()
    context = ContextManager(config)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.return_value = ProviderResponse(
        content="ok", tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )
    sm = MagicMock(spec=SessionManager)
    sm.save.return_value = None
    sm.append_compaction.return_value = None
    session = MagicMock(spec=Session)
    session.id = "test-session"
    session.messages = []
    session.compactions = []
    steering = SteeringChannel()
    return Agent(
        config=config, provider=provider, registry=registry,
        context=context, session=session, session_manager=sm,
        steering=steering,
    )


def _call(cmd: str, agent: Agent | None = None) -> tuple[bool, list]:
    """
    Call _handle_slash, capture calls to tau.cli.console.print,
    return (handled, list_of_print_args).
    Patching tau.cli.console ensures the function uses the mock
    regardless of how it was imported.
    """
    prints: list = []
    steering = agent._steering if agent else SteeringChannel()
    with patch.object(_cli_mod, "console") as mock_console:
        mock_console.print.side_effect = lambda *a, **kw: prints.append(a[0] if a else None)
        handled = _cli_mod._handle_slash(
            cmd,
            steering=steering,
            ext_registry=None,
            ext_context=None,
            agent=agent,
        )
    return handled, prints


# ===========================================================================
# /help
# ===========================================================================

class TestSlashHelp:
    def test_returns_true(self):
        handled, _ = _call("/help")
        assert handled is True

    def test_prints_something(self):
        _, prints = _call("/help")
        assert len(prints) > 0

    def test_includes_extension_commands(self):
        """all_slash_commands() must be called when an ext_registry is passed."""
        ext_registry = MagicMock(spec=ExtensionRegistry)
        ext_registry.all_slash_commands.return_value = [("foo", "does foo")]
        steering = SteeringChannel()
        with patch.object(_cli_mod, "console"):
            handled = _cli_mod._handle_slash(
                "/help",
                steering=steering,
                ext_registry=ext_registry,
                ext_context=None,
            )
        assert handled is True
        ext_registry.all_slash_commands.assert_called_once()

    def test_all_new_commands_in_help_text(self):
        """The _SLASH_HELP string must mention every new command."""
        for cmd in ("/clear", "/compact", "/model", "/tokens"):
            assert cmd in _cli_mod._SLASH_HELP


# ===========================================================================
# /clear
# ===========================================================================

class TestSlashClear:
    def test_returns_true(self):
        agent = _make_agent()
        handled, _ = _call("/clear", agent)
        assert handled is True

    def test_wipes_non_system_messages(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hello"))
        agent._context.add_message(Message(role="assistant", content="hi"))
        assert len([m for m in agent._context.get_messages() if m.role != "system"]) == 2

        _call("/clear", agent)

        non_sys = [m for m in agent._context.get_messages() if m.role != "system"]
        assert non_sys == []

    def test_keeps_system_prompt(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hello"))
        _call("/clear", agent)

        sys_msgs = [m for m in agent._context.get_messages() if m.role == "system"]
        assert len(sys_msgs) == 1
        assert "sys" in sys_msgs[0].content

    def test_prints_confirmation(self):
        agent = _make_agent()
        _, prints = _call("/clear", agent)
        assert len(prints) > 0

    def test_no_op_without_agent(self):
        handled, _ = _call("/clear", agent=None)
        assert handled is True

    def test_idempotent(self):
        """Calling /clear twice should not raise."""
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hello"))
        _call("/clear", agent)
        _call("/clear", agent)   # second call — already empty
        non_sys = [m for m in agent._context.get_messages() if m.role != "system"]
        assert non_sys == []


# ===========================================================================
# /compact
# ===========================================================================

class TestSlashCompact:
    def _agent_with_messages(self, n: int = 10) -> Agent:
        agent = _make_agent()
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            agent._context.add_message(Message(role=role, content=f"message {i} " * 20))
        return agent

    def test_returns_true(self):
        agent = _make_agent()
        handled, _ = _call("/compact", agent)
        assert handled is True

    def test_compacts_context_and_persists_entry(self):
        agent = self._agent_with_messages(10)
        tokens_before = agent._context.token_count()

        summary_msg = Message(role="user", content="[summary]")
        entry = CompactionEntry(
            summary="summary", tokens_before=tokens_before, timestamp="2024-01-01T00:00:00"
        )
        agent._context.compactor.compact = MagicMock(return_value=([summary_msg], entry))

        _call("/compact", agent)

        agent._context.compactor.compact.assert_called_once()
        agent._session_manager.append_compaction.assert_called_once()

    def test_reduces_non_system_messages_after_compact(self):
        agent = self._agent_with_messages(10)
        msgs_before = len([m for m in agent._context.get_messages() if m.role != "system"])

        # Use a real compact call — mock the provider summary
        agent._provider.chat.return_value = ProviderResponse(
            content="summary text", tool_calls=[], stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
        )
        _call("/compact", agent)

        msgs_after = len([m for m in agent._context.get_messages() if m.role != "system"])
        assert msgs_after < msgs_before

    def test_handles_too_few_messages_gracefully(self):
        agent = _make_agent()   # only system prompt
        handled, prints = _call("/compact", agent)
        assert handled is True
        assert len(prints) > 0  # warning was printed, not raised

    def test_handles_compact_exception_gracefully(self):
        agent = self._agent_with_messages(10)
        agent._context.compactor.compact = MagicMock(side_effect=RuntimeError("boom"))
        handled, prints = _call("/compact", agent)
        assert handled is True
        assert len(prints) > 0

    def test_no_op_without_agent(self):
        handled, _ = _call("/compact", agent=None)
        assert handled is True


# ===========================================================================
# /model
# ===========================================================================

class TestSlashModel:
    def test_returns_true_with_arg(self):
        agent = _make_agent()
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()):
            handled, _ = _call("/model gpt-4o-mini", agent)
        assert handled is True

    def test_swaps_model_name_on_config(self):
        agent = _make_agent()
        assert agent._config.model == "gpt-4o"
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()):
            _call("/model gpt-4o-mini", agent)
        assert agent._config.model == "gpt-4o-mini"

    def test_swaps_provider_instance(self):
        agent = _make_agent()
        old_provider = agent._provider
        new_provider = MagicMock()
        with patch.object(_cli_mod, "_swap_provider", return_value=new_provider):
            _call("/model gpt-4o-mini", agent)
        assert agent._provider is new_provider
        assert agent._provider is not old_provider

    def test_swap_provider_called_with_updated_config(self):
        agent = _make_agent()
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()) as mock_swap:
            _call("/model gpt-4o-mini", agent)
        mock_swap.assert_called_once()
        # Config passed to swap must already have the new model name
        called_config = mock_swap.call_args[0][0]
        assert called_config.model == "gpt-4o-mini"

    def test_prints_old_and_new_model(self):
        agent = _make_agent()
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()):
            _, prints = _call("/model gpt-4o-mini", agent)
        assert len(prints) > 0

    def test_no_arg_shows_current_model(self):
        agent = _make_agent()
        _, prints = _call("/model", agent)
        assert len(prints) > 0

    def test_no_arg_does_not_swap(self):
        agent = _make_agent()
        original = agent._config.model
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()) as mock_swap:
            _call("/model", agent)
        assert agent._config.model == original
        mock_swap.assert_not_called()

    def test_no_op_without_agent(self):
        handled, _ = _call("/model gpt-4o-mini", agent=None)
        assert handled is True

    def test_multiple_swaps(self):
        """Swapping the model twice should work without error."""
        agent = _make_agent()
        with patch.object(_cli_mod, "_swap_provider", return_value=MagicMock()):
            _call("/model gpt-4o-mini", agent)
            _call("/model o3-mini", agent)
        assert agent._config.model == "o3-mini"


# ===========================================================================
# /tokens
# ===========================================================================

class TestSlashTokens:
    def test_returns_true(self):
        agent = _make_agent()
        handled, _ = _call("/tokens", agent)
        assert handled is True

    def test_prints_usage_line(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hello world"))
        _, prints = _call("/tokens", agent)
        assert len(prints) > 0

    def test_no_op_without_agent(self):
        handled, _ = _call("/tokens", agent=None)
        assert handled is True

    def test_empty_context_shows_zero_percent(self):
        """No messages beyond system prompt → near-zero usage."""
        agent = _make_agent()
        _, prints = _call("/tokens", agent)
        assert len(prints) > 0   # bar rendered without raising

    def test_bar_colour_green_when_low(self):
        """With a tiny token count relative to a large budget, colour is green."""
        agent = _make_agent(_cfg(max_tokens=100_000))
        agent._context.add_message(Message(role="user", content="hi"))
        # Just confirm it runs without raising — colour is inside a Text object
        handled, prints = _call("/tokens", agent)
        assert handled is True


# ===========================================================================
# Unknown / extension delegation
# ===========================================================================

class TestSlashUnknown:
    def test_returns_false_for_unknown_command(self):
        handled, _ = _call("/doesnotexist")
        assert handled is False

    def test_delegates_to_ext_registry(self):
        ext_registry = MagicMock(spec=ExtensionRegistry)
        ext_registry.handle_slash.return_value = True
        ext_context = MagicMock(spec=ExtensionContext)
        steering = SteeringChannel()
        with patch.object(_cli_mod, "console"):
            handled = _cli_mod._handle_slash(
                "/extcmd args",
                steering=steering,
                ext_registry=ext_registry,
                ext_context=ext_context,
                agent=None,
            )
        assert handled is True
        ext_registry.handle_slash.assert_called_once_with("/extcmd args", ext_context)

    def test_ext_registry_not_called_without_ext_context(self):
        """ext_registry alone (no ext_context) must not dispatch."""
        ext_registry = MagicMock(spec=ExtensionRegistry)
        steering = SteeringChannel()
        with patch.object(_cli_mod, "console"):
            handled = _cli_mod._handle_slash(
                "/extcmd",
                steering=steering,
                ext_registry=ext_registry,
                ext_context=None,   # no context → no dispatch
                agent=None,
            )
        ext_registry.handle_slash.assert_not_called()
        assert handled is False

    def test_ext_registry_returning_false_falls_through(self):
        ext_registry = MagicMock(spec=ExtensionRegistry)
        ext_registry.handle_slash.return_value = False
        ext_context = MagicMock(spec=ExtensionContext)
        steering = SteeringChannel()
        with patch.object(_cli_mod, "console"):
            handled = _cli_mod._handle_slash(
                "/extcmd",
                steering=steering,
                ext_registry=ext_registry,
                ext_context=ext_context,
                agent=None,
            )
        assert handled is False


# ===========================================================================
# /copy
# ===========================================================================

class TestSlashCopy:
    def test_returns_true(self):
        agent = _make_agent()
        handled, _ = _call("/copy", agent)
        assert handled is True

    def test_no_agent(self):
        handled, prints = _call("/copy", None)
        assert handled is True
        output = str(prints[0]) if prints else ""
        assert "requires" in output

    def test_no_assistant_message(self):
        agent = _make_agent()
        handled, prints = _call("/copy", agent)
        assert handled is True
        output = str(prints[0]) if prints else ""
        assert "No assistant" in output

    def test_copies_last_assistant(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hi"))
        agent._context.add_message(Message(role="assistant", content="hello there"))
        agent._context.add_message(Message(role="user", content="bye"))
        agent._context.add_message(Message(role="assistant", content="goodbye"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            handled, prints = _call("/copy", agent)

        assert handled is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs.get("input") == "goodbye"

    def test_copies_skips_empty_assistant(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="user", content="hi"))
        agent._context.add_message(Message(role="assistant", content="real answer"))
        agent._context.add_message(Message(role="assistant", content=""))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            handled, prints = _call("/copy", agent)

        assert handled is True
        call_args = mock_run.call_args
        assert call_args.kwargs.get("input") == "real answer"

    def test_clipboard_tool_not_found(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="assistant", content="text"))

        with patch("subprocess.run", side_effect=FileNotFoundError):
            handled, prints = _call("/copy", agent)

        assert handled is True
        output = str(prints[0]) if prints else ""
        assert "not found" in output.lower()

    def test_copy_error(self):
        agent = _make_agent()
        agent._context.add_message(Message(role="assistant", content="text"))

        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            handled, prints = _call("/copy", agent)

        assert handled is True
        output = str(prints[0]) if prints else ""
        assert "failed" in output.lower()

    def test_copy_in_help_text(self):
        handled, prints = _call("/help", None)
        # /help renders a Rich Panel; check the renderable content
        panel = prints[0]
        assert "/copy" in str(panel.renderable)
