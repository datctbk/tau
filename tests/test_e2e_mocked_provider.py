"""End-to-end integration test using real storage managers and a mock provider.

Validates the full user_input -> agent.run() -> tool execution -> response loop.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Generator

import pytest

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    Message,
    ProviderResponse,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallEvent,
    ToolDefinition,
    ToolParameter,
    ToolResultEvent,
)
from tau.providers.base import BaseProvider


class E2EMockProvider(BaseProvider):
    """A mock provider that implements the BaseProvider protocol properly
    and yields pre-programmed conversational turns.
    """
    def __init__(self, turns: list[ProviderResponse]):
        self._name = "mock_provider"
        self._turns = turns
        self._turn_index = 0
        self.last_response_headers = {"x-ratelimit-remaining-requests": "100"}
        self.recorded_requests = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: list[Message], tools: list[ToolDefinition], stream: bool = True) -> ProviderResponse | Generator:
        self.recorded_requests.append((messages, tools))
        
        if self._turn_index >= len(self._turns):
            raise RuntimeError("Provider called more times than programmed turns.")
            
        current_turn = self._turns[self._turn_index]
        self._turn_index += 1
        
        if stream:
            return self._stream_response(current_turn)
        else:
            return current_turn
            
    def _stream_response(self, response: ProviderResponse) -> Generator:
        # Simulate text stream if there is content
        if response.content:
            words = response.content.split()
            for word in words:
                yield TextDelta(text=word + " ")
                
        # Then simulate the tool call / usage completion
        yield response
        
    import contextlib
    @contextlib.contextmanager
    def swap_model(self, model: str):
        yield


@pytest.fixture
def temp_tau_home():
    """Create a temporary TAU_HOME directory for real database files."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "tau.db"
    yield db_path
    shutil.rmtree(temp_dir)


def test_agent_e2e_full_loop(temp_tau_home):
    """Test the complete core loop: initial prompt -> tool call -> tool result -> final answer."""
    # 1. Setup real state dependencies
    config = AgentConfig(model="mock-gpt", provider="mock")
    session_manager = SessionManager(sessions_dir=Path(temp_tau_home))
    session = session_manager.new_session(config, name="e2e-test")
    context = ContextManager(config)
    registry = ToolRegistry()
    
    # 2. Register a real callable tool
    def get_weather(location: str) -> str:
        return f"The weather in {location} is 72F and sunny."
        
    tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": ToolParameter(type="string", description="City name")
        },
        handler=get_weather,
    )
    registry.register(tool)
    
    # 3. Setup Mock Provider Responses
    # Turn 1: LLM decides to call the get_weather tool
    turn_1 = ProviderResponse(
        content="I will check the weather for you.",
        tool_calls=[ToolCall(id="call_abc123", name="get_weather", arguments={"location": "London"})],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=10, output_tokens=15)
    )
    # Turn 2: LLM receives tool result and provides final answer
    turn_2 = ProviderResponse(
        content="It is currently 72F and sunny in London.",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=50, output_tokens=10)
    )
    mock_provider = E2EMockProvider([turn_1, turn_2])
    
    # 4. Initialize Agent
    agent = Agent(
        config=config,
        provider=mock_provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=session_manager,
    )
    
    # 5. Run the agent and collect events
    events = list(agent.run("What is the weather in London?"))
    
    # Verify Events Emitted Correctly
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    
    assert len(tool_calls) == 1
    assert tool_calls[0].call.name == "get_weather"
    assert tool_calls[0].call.arguments == {"location": "London"}
    
    assert len(tool_results) == 1
    assert tool_results[0].result.content == "The weather in London is 72F and sunny."
    
    # Check that text was streamed
    assert len(text_deltas) > 0
    full_emitted_text = "".join(d.text for d in text_deltas)
    assert "72F and sunny" in full_emitted_text
    
    # 6. Verify Session Storage State
    # Reload session from DB to verify persistence occurred successfully
    reloaded_session = session_manager.load(session.id)
    assert reloaded_session is not None
    
    # Expected messages:
    # 1. User: "What is the weather in London?"
    # 2. Assistant: (tool call msg)
    # 3. Tool: (tool result msg)
    # 4. Assistant: "It is currently 72F and sunny..."
    saved_messages = reloaded_session.messages
    assert len(saved_messages) == 5
    assert saved_messages[0]["role"] == "system"
    assert saved_messages[1]["role"] == "user"
    assert saved_messages[2]["role"] == "assistant"
    assert saved_messages[2]["tool_calls"][0]["name"] == "get_weather"
    assert saved_messages[3]["role"] == "tool"
    assert saved_messages[4]["role"] == "assistant"
    assert "72F and sunny" in saved_messages[4]["content"]
