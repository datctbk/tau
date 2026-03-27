"""Tests for --max-cost budget guard and --no-session ephemeral mode."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    CostLimitExceeded,
    ErrorEvent,
    ProviderResponse,
    TextChunk,
    TokenUsage,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolResultEvent,
    TurnComplete,
)
from tau.sdk import InMemorySessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> AgentConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        max_turns=10,
        max_tokens=8192,
        compaction_enabled=False,
        retry_enabled=False,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _make_agent(
    provider_responses: list,
    config: AgentConfig | None = None,
    extra_tools: list[ToolDefinition] | None = None,
    cost_calculator=None,
) -> Agent:
    cfg = config or _cfg()
    context = ContextManager(cfg)
    registry = ToolRegistry()
    if extra_tools:
        registry.register_many(extra_tools)

    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    session = MagicMock()
    session.id = "test-session"
    session.messages = []
    session.cumulative_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    sm = MagicMock()
    sm.save.return_value = None

    return Agent(
        config=cfg,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
        cost_calculator=cost_calculator,
    )


# ===========================================================================
# CostLimitExceeded event
# ===========================================================================


class TestCostLimitExceeded:
    def test_to_dict(self):
        evt = CostLimitExceeded(session_cost=0.123, max_cost=0.100)
        d = evt.to_dict()
        assert d["type"] == "cost_limit_exceeded"
        assert d["session_cost"] == 0.123
        assert d["max_cost"] == 0.100

    def test_fields(self):
        evt = CostLimitExceeded(session_cost=0.5, max_cost=0.3)
        assert evt.session_cost == 0.5
        assert evt.max_cost == 0.3


# ===========================================================================
# Budget guard in agent loop
# ===========================================================================


class TestBudgetGuard:
    def test_stops_when_cost_exceeds_budget(self):
        """Agent should yield CostLimitExceeded and stop when session cost >= max_cost."""
        # cost_calculator returns $0.50 per call
        calc = lambda model, usage: 0.50

        cfg = _cfg(max_cost=0.10)
        agent = _make_agent(
            provider_responses=[
                ProviderResponse(
                    content="Hello!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                ),
            ],
            config=cfg,
            cost_calculator=calc,
        )
        events = list(agent.run("hi"))
        # Should have TurnComplete then CostLimitExceeded
        types = [type(e).__name__ for e in events]
        assert "TurnComplete" in types
        assert "CostLimitExceeded" in types
        evt = next(e for e in events if isinstance(e, CostLimitExceeded))
        assert evt.session_cost == 0.50
        assert evt.max_cost == 0.10

    def test_no_stop_when_under_budget(self):
        """Agent proceeds normally when cost is under budget."""
        calc = lambda model, usage: 0.01

        cfg = _cfg(max_cost=1.00)
        agent = _make_agent(
            provider_responses=[
                ProviderResponse(
                    content="Hello!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                ),
            ],
            config=cfg,
            cost_calculator=calc,
        )
        events = list(agent.run("hi"))
        assert not any(isinstance(e, CostLimitExceeded) for e in events)
        assert any(isinstance(e, TurnComplete) for e in events)
        assert any(isinstance(e, TextChunk) for e in events)

    def test_no_check_when_max_cost_zero(self):
        """Budget guard is disabled when max_cost == 0 (unlimited)."""
        # Would exceed any budget, but max_cost=0 means disabled
        calc = lambda model, usage: 999.0

        cfg = _cfg(max_cost=0.0)
        agent = _make_agent(
            provider_responses=[
                ProviderResponse(
                    content="Still going!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                ),
            ],
            config=cfg,
            cost_calculator=calc,
        )
        events = list(agent.run("hi"))
        assert not any(isinstance(e, CostLimitExceeded) for e in events)

    def test_no_check_when_no_calculator(self):
        """Budget guard does nothing if cost_calculator not provided."""
        cfg = _cfg(max_cost=0.01)
        agent = _make_agent(
            provider_responses=[
                ProviderResponse(
                    content="No calc!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=10000, output_tokens=5000),
                ),
            ],
            config=cfg,
            cost_calculator=None,
        )
        events = list(agent.run("hi"))
        assert not any(isinstance(e, CostLimitExceeded) for e in events)

    def test_stops_mid_tool_loop(self):
        """Budget exceeded during a tool-call loop halts before the next tool dispatch."""
        call_count = [0]

        def expensive_calc(model, usage):
            call_count[0] += 1
            # First turn: $0.05 (under), second turn: $0.20 (over)
            return 0.05 * call_count[0]

        call = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
        tool = ToolDefinition(
            name="echo",
            description="echo",
            parameters={"text": ToolParameter(type="string", description="text")},
            handler=lambda text: text,
        )
        cfg = _cfg(max_cost=0.08)
        agent = _make_agent(
            provider_responses=[
                # Turn 1: tool call → under budget
                ProviderResponse(
                    content=None,
                    tool_calls=[call],
                    stop_reason="tool_use",
                    usage=TokenUsage(input_tokens=50, output_tokens=20),
                ),
                # Turn 2: text response → over budget at this point
                ProviderResponse(
                    content="Done!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=60, output_tokens=30),
                ),
            ],
            config=cfg,
            extra_tools=[tool],
            cost_calculator=expensive_calc,
        )
        events = list(agent.run("echo hi"))
        assert any(isinstance(e, CostLimitExceeded) for e in events)

    def test_persist_called_on_budget_exceeded(self):
        """Session is persisted when budget is exceeded."""
        calc = lambda model, usage: 1.00

        cfg = _cfg(max_cost=0.50)
        agent = _make_agent(
            provider_responses=[
                ProviderResponse(
                    content="Expensive!",
                    tool_calls=[],
                    stop_reason="end_turn",
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                ),
            ],
            config=cfg,
            cost_calculator=calc,
        )
        events = list(agent.run("hi"))
        assert any(isinstance(e, CostLimitExceeded) for e in events)
        # _persist() calls session_manager.save()
        agent._session_manager.save.assert_called()


# ===========================================================================
# AgentConfig.max_cost field
# ===========================================================================


class TestAgentConfigMaxCost:
    def test_default_is_zero(self):
        cfg = AgentConfig()
        assert cfg.max_cost == 0.0

    def test_set_via_constructor(self):
        cfg = AgentConfig(max_cost=5.0)
        assert cfg.max_cost == 5.0


# ===========================================================================
# --no-session (ephemeral mode) via CLI wiring
# ===========================================================================


class TestNoSessionCLI:
    def test_no_session_flag_uses_in_memory_session_manager(self):
        """When --no-session is set, run_cmd should use InMemorySessionManager."""
        import tau.cli as cli_mod

        with patch.object(cli_mod, "_setup_logging"), \
             patch.object(cli_mod, "ensure_tau_home"), \
             patch.object(cli_mod, "load_config") as mock_load, \
             patch.object(cli_mod, "_make_agent_config") as mock_make, \
             patch.object(cli_mod, "_build_agent") as mock_build, \
             patch.object(cli_mod, "_render_events") as mock_render, \
             patch("tau.cli.theme"):

            mock_load.return_value = MagicMock()
            mock_make.return_value = _cfg()
            mock_agent = MagicMock()
            mock_ext = MagicMock()
            mock_build.return_value = (mock_agent, mock_ext)

            # Call the underlying function (not the Click command)
            cli_mod.run_cmd.callback(
                prompt="hello",
                provider=None, model=None, think=None,
                image=(), resume_id=None, session_name=None,
                no_confirm=False, no_parallel=False, persistent_shell=False,
                workspace=".", verbose=False, show_thinking=False, output_mode=None,
                print_mode=False, template_name=None, var=(),
                max_cost=None, no_session=True, trace_log=None,
            )

            # Check that _build_agent was called with an InMemorySessionManager
            call_kwargs = mock_build.call_args
            sm = call_kwargs.kwargs.get("session_manager") or call_kwargs[1].get("session_manager")
            if sm is None:
                # positional args: tau_config, agent_config, session_manager, ...
                sm = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
            assert isinstance(sm, InMemorySessionManager)

    def test_default_uses_disk_session_manager(self):
        """When --no-session is not set, run_cmd should use SessionManager (disk)."""
        import tau.cli as cli_mod

        with patch.object(cli_mod, "_setup_logging"), \
             patch.object(cli_mod, "ensure_tau_home"), \
             patch.object(cli_mod, "load_config") as mock_load, \
             patch.object(cli_mod, "_make_agent_config") as mock_make, \
             patch.object(cli_mod, "_build_agent") as mock_build, \
             patch.object(cli_mod, "_render_events") as mock_render, \
             patch("tau.cli.theme"):

            mock_load.return_value = MagicMock()
            mock_make.return_value = _cfg()
            mock_agent = MagicMock()
            mock_ext = MagicMock()
            mock_build.return_value = (mock_agent, mock_ext)

            cli_mod.run_cmd.callback(
                prompt="hello",
                provider=None, model=None, think=None,
                image=(), resume_id=None, session_name=None,
                no_confirm=False, no_parallel=False, persistent_shell=False,
                workspace=".", verbose=False, show_thinking=False, output_mode=None,
                print_mode=False, template_name=None, var=(),
                max_cost=None, no_session=False, trace_log=None,
            )

            call_kwargs = mock_build.call_args
            sm = call_kwargs.kwargs.get("session_manager") or call_kwargs[1].get("session_manager")
            if sm is None:
                sm = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
            assert isinstance(sm, SessionManager)
            assert not isinstance(sm, InMemorySessionManager)


# ===========================================================================
# --max-cost CLI wiring
# ===========================================================================


class TestMaxCostCLI:
    def test_max_cost_passed_to_agent_config(self):
        """--max-cost value should land in AgentConfig.max_cost."""
        from tau.config import TauConfig
        import tau.cli as cli_mod

        tau_config = TauConfig()
        cfg = cli_mod._make_agent_config(
            tau_config,
            provider=None, model=None, think=None,
            no_confirm=False, workspace=".",
            max_cost=2.50,
        )
        assert cfg.max_cost == 2.50

    def test_max_cost_none_falls_back_to_config(self):
        """When --max-cost is not given, uses TauConfig.max_cost."""
        from tau.config import TauConfig
        import tau.cli as cli_mod

        tau_config = TauConfig(max_cost=1.23)
        cfg = cli_mod._make_agent_config(
            tau_config,
            provider=None, model=None, think=None,
            no_confirm=False, workspace=".",
            max_cost=None,
        )
        assert cfg.max_cost == 1.23

    def test_max_cost_default_is_zero(self):
        """Default max_cost is 0 (unlimited)."""
        from tau.config import TauConfig
        import tau.cli as cli_mod

        tau_config = TauConfig()
        cfg = cli_mod._make_agent_config(
            tau_config,
            provider=None, model=None, think=None,
            no_confirm=False, workspace=".",
        )
        assert cfg.max_cost == 0.0

    def test_cost_calculator_passed_to_agent(self):
        """_build_agent should pass tau_config.calculate_cost as cost_calculator."""
        import tau.cli as cli_mod

        with patch.object(cli_mod, "register_builtin_tools"), \
             patch.object(cli_mod, "configure_shell"), \
             patch.object(cli_mod, "configure_fs"), \
             patch.object(cli_mod, "configure_context"), \
             patch.object(cli_mod, "get_provider") as mock_prov, \
             patch("tau.context_files.load_system_prompt_override", return_value=None), \
             patch("tau.context_files.load_context_files", return_value=""):

            mock_prov.return_value = MagicMock()
            tau_config = MagicMock()
            tau_config.tools.enabled_only = []
            tau_config.tools.disabled = []
            tau_config.shell = MagicMock()
            tau_config.ollama = MagicMock()
            tau_config.skills.paths = []
            tau_config.skills.disabled = []
            tau_config.extensions.paths = []
            tau_config.extensions.disabled = []
            tau_config.calculate_cost = MagicMock(return_value=0.01)

            sm = InMemorySessionManager()
            cfg = _cfg()
            agent, _ = cli_mod._build_agent(
                tau_config=tau_config,
                agent_config=cfg,
                session_manager=sm,
                session_name=None,
                resume_id=None,
            )
            assert agent._cost_calculator is tau_config.calculate_cost
