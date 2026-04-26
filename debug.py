from unittest.mock import MagicMock
from tau.core.agent import Agent
from tau.core.types import AgentConfig, Message, ProviderResponse, TokenUsage
from tau.core.usage_pricing import CanonicalUsage, estimate_usage_cost
from tests.test_agent_p1_wiring import MockP1Provider

session = MagicMock()
session.cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}

cu = getattr(session, "cumulative_usage", None)
cu["input_tokens"] += 1_000_000
cu["output_tokens"] += 1_000_000

print(cu)
c_usage = CanonicalUsage(**cu)
cost = estimate_usage_cost("gpt-4o", c_usage)
print(cost)
