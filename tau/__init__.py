"""tau — a minimal, extensible CLI coding agent."""

__version__ = "0.1.0"

# Public SDK API
from tau.sdk import TauSession, InMemorySessionManager, create_session  # noqa: F401
from tau.rpc import run_rpc, start_rpc  # noqa: F401
