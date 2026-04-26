"""Tests for P1 infrastructure wired into the agent loop."""

import pytest
from unittest.mock import MagicMock, patch

from tau.core.agent import Agent
from tau.core.types import AgentConfig, Message, ProviderResponse, TokenUsage, TurnComplete, TextDelta, CostLimitExceeded
from tau.providers.base import BaseProvider
from tau.core.rate_limit_tracker import RateLimitState

class MockP1Provider(BaseProvider):
    def __init__(self, name="openai"):
        self._name = name
        self.last_response_headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        self.swapped_models = []
        self._chat_return = ProviderResponse(content="Hello", tool_calls=[], stop_reason="end_turn", usage=TokenUsage(100, 50, 0, 0))

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, tools, stream=True):
        self.last_messages = messages
        if stream:
            yield TextDelta(text="Hello")
            yield self._chat_return
        else:
            return self._chat_return

    import contextlib
    @contextlib.contextmanager
    def swap_model(self, model: str):
        old = getattr(self, "current_model", None)
        self.swapped_models.append(model)
        self.current_model = model
        try:
            yield
        finally:
            self.current_model = old

@pytest.fixture
def mock_agent_deps():
    config = AgentConfig(model="gpt-4o", provider="openai")
    registry = MagicMock()
    registry.all_definitions.return_value = []
    
    context = MagicMock()
    context.get_messages.return_value = [Message(role="user", content="Hi")]
    context.token_count.return_value = 10
    
    compactor = MagicMock()
    compactor.should_compact.return_value = False
    context.compactor = compactor
    
    session = MagicMock()
    session.cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    session_manager = MagicMock()
    
    return config, registry, context, session, session_manager

def test_rate_limit_tracking(mock_agent_deps):
    config, registry, context, session, session_manager = mock_agent_deps
    provider = MockP1Provider(name="openai")
    
    agent = Agent(config, provider, registry, context, session, session_manager)
    
    # Run one turn
    events = list(agent.run("Hi"))
    
    # Verify rate limit state was parsed and stored
    assert hasattr(session, "rate_limit_state")
    rl = session.rate_limit_state
    assert isinstance(rl, RateLimitState)
    assert rl.requests_min.limit == 100
    assert rl.requests_min.remaining == 99

def test_smart_routing(mock_agent_deps):
    config, registry, context, session, session_manager = mock_agent_deps
    class DummyRouting:
        def to_routing_dict(self):
            return {
                "enabled": True,
                "max_simple_chars": 160,
                "max_simple_words": 28,
                "cheap_model": {"provider": "openai", "model": "gpt-4o-mini"}
            }
    config.smart_routing_config = DummyRouting()
    provider = MockP1Provider(name="openai")
    
    agent = Agent(config, provider, registry, context, session, session_manager)
    
    list(agent.run("Hi there!"))  # Very simple message
    
    # Provider should have swapped to cheap model
    assert "gpt-4o-mini" in provider.swapped_models

def test_smart_routing_complex_message_keeps_primary(mock_agent_deps):
    config, registry, context, session, session_manager = mock_agent_deps
    class DummyRouting:
        def to_routing_dict(self):
            return {
                "enabled": True,
                "max_simple_chars": 160,
                "max_simple_words": 28,
                "cheap_model": {"provider": "openai", "model": "gpt-4o-mini"}
            }
    config.smart_routing_config = DummyRouting()
    provider = MockP1Provider(name="openai")
    
    agent = Agent(config, provider, registry, context, session, session_manager)
    
    list(agent.run("Please debug this python exception trace."))  # Complex keyword
    
    # Provider should NOT have swapped
    assert len(provider.swapped_models) == 0

def test_usage_pricing_enforcement(mock_agent_deps):
    config, registry, context, session, session_manager = mock_agent_deps
    config.max_cost = 0.0001  # Very low budget
    provider = MockP1Provider(name="openai")
    
    # Mock return 1,000,000 tokens to blow the budget immediately
    provider._chat_return = ProviderResponse(
        content="Expensive", 
        tool_calls=[], 
        stop_reason="end_turn", 
        usage=TokenUsage(1_000_000, 1_000_000, 0, 0)
    )
    
    agent = Agent(config, provider, registry, context, session, session_manager)
    events = list(agent.run("Hi"))
    
    # Should contain CostLimitExceeded event
    assert any(isinstance(e, CostLimitExceeded) for e in events)

def test_prompt_caching(mock_agent_deps):
    config, registry, context, session, session_manager = mock_agent_deps
    config.model = "claude-3-5-sonnet-20241022"  # Triggers Anthropic cache logic
    provider = MockP1Provider(name="openai")
    
    agent = Agent(config, provider, registry, context, session, session_manager)
    list(agent.run("Hi"))
    
    # Verify cache_control was injected
    assert len(provider.last_messages) > 0
    msg_dict = provider.last_messages[-1].to_dict()
    # User message text transformed to list of dicts with cache_control
    assert isinstance(msg_dict["content"], list)
    assert msg_dict["content"][0]["cache_control"] == {"type": "ephemeral"}
