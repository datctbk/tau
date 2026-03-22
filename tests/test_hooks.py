import pytest

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.extension import Extension, ExtensionRegistry
from tau.core.session import Session, SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    ToolResultEvent,
)
from unittest.mock import MagicMock


class BlockingExtension(Extension):
    # Minimal manifest to satisfy requirements
    from tau.core.types import ExtensionManifest
    manifest = ExtensionManifest(name="blocking_ext", description="Blocks tools")

    def before_tool_call(self, context: BeforeToolCallContext) -> BeforeToolCallResult | None:
        if context.tool_call.name == "forbidden_tool":
            return BeforeToolCallResult(block=True, reason="ACCESS DENIED")
        return None

    def after_tool_call(self, context: AfterToolCallContext) -> AfterToolCallResult | None:
        if context.tool_call.name == "allowed_tool":
            return AfterToolCallResult(content="Intercepted and modified")
        return None


def test_agent_hooks():
    config = AgentConfig(max_turns=1)
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="forbidden_tool", description="", parameters={"type": "object"}, handler=lambda: "success"))
    registry.register(ToolDefinition(name="allowed_tool", description="", parameters={"type": "object"}, handler=lambda: "success"))

    provider = MagicMock()
    # Provider will yield tool calls
    provider.chat.return_value = ProviderResponse(
        content=None,
        tool_calls=[
            ToolCall(id="tc1", name="forbidden_tool", arguments={}),
            ToolCall(id="tc2", name="allowed_tool", arguments={}),
        ],
        stop_reason="tool_use",
    )

    context = ContextManager(config)
    session = MagicMock()
    session.id = "test"
    session.messages = []
    session_manager = MagicMock()
    session_manager.new_session.return_value = session
    session_manager.save.return_value = None

    ext_registry = ExtensionRegistry(include_builtins=False)
    # Manually register the test extension
    ext = BlockingExtension()
    ext_registry._extensions[ext.manifest.name] = ext

    agent = Agent(
        config=config,
        provider=provider,
        registry=registry,
        ext_registry=ext_registry,
        context=context,
        session=session,
        session_manager=session_manager,
    )

    events = list(agent.run("do evil stuff"))

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 2

    # First tool should be blocked
    assert tool_results[0].result.tool_call_id == "tc1"
    assert tool_results[0].result.is_error is True
    assert tool_results[0].result.content == "ACCESS DENIED"

    # Second tool should be intercepted and modified
    assert tool_results[1].result.tool_call_id == "tc2"
    assert tool_results[1].result.content == "Intercepted and modified"
