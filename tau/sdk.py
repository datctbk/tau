"""tau SDK — programmatic API for embedding tau in other applications."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from tau.config import TauConfig, ensure_tau_home, load_config
from tau.core.agent import Agent
from tau.core.context import ContextManager, configure_context
from tau.core.extension import ExtensionRegistry
from tau.core.session import Session, SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig, Event
from tau.providers import get_provider
from tau.skills import SkillLoader
from tau.tools import register_builtin_tools
from tau.tools.fs import configure_fs
from tau.tools.shell import configure_shell

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory session manager (no disk I/O)
# ---------------------------------------------------------------------------


class InMemorySessionManager(SessionManager):
    """A session manager that stores everything in memory — no file system."""

    def __init__(self) -> None:
        # Do NOT call super().__init__() which creates dirs on disk.
        self._store: dict[str, dict[str, Any]] = {}

    def _path(self, session_id: str) -> Path:
        raise NotImplementedError("InMemorySessionManager has no file paths")

    def new_session(self, config: AgentConfig, name: str | None = None) -> Session:
        now = datetime.now(tz=datetime.now().astimezone().tzinfo).isoformat()
        session = Session(
            id=str(uuid.uuid4()),
            name=name,
            created_at=now,
            updated_at=now,
            config=config,
        )
        self._store[session.id] = session.to_dict()
        return session

    def save(self, session: Session, messages: list[dict] | None = None) -> None:
        if messages is not None:
            session.messages = messages
        session.updated_at = datetime.now(
            tz=datetime.now().astimezone().tzinfo
        ).isoformat()
        self._store[session.id] = session.to_dict()

    def load(self, session_id: str) -> Session:
        if session_id in self._store:
            return Session.from_dict(self._store[session_id])
        # Try prefix match
        matches = [
            sid for sid in self._store if sid.startswith(session_id)
        ]
        if len(matches) == 1:
            return Session.from_dict(self._store[matches[0]])
        if len(matches) > 1:
            from tau.core.session import SessionNotFoundError
            raise SessionNotFoundError(
                f"Ambiguous session prefix {session_id!r}"
            )
        from tau.core.session import SessionNotFoundError
        raise SessionNotFoundError(f"Session {session_id!r} not found.")

    def list_sessions(self):
        return [
            Session.from_dict(d).meta
            for d in self._store.values()
        ]

    def delete(self, session_id: str) -> None:
        if session_id not in self._store:
            from tau.core.session import SessionNotFoundError
            raise SessionNotFoundError(f"Session {session_id!r} not found.")
        del self._store[session_id]

    def append_compaction(self, session, entry) -> None:
        session.compactions.append({
            "summary": entry.summary,
            "tokens_before": entry.tokens_before,
            "timestamp": entry.timestamp,
        })
        self.save(session)

    def fork(self, session_id, fork_index, name=None):
        parent = self.load(session_id)
        if fork_index < 0 or fork_index >= len(parent.messages):
            raise ValueError(f"fork_index {fork_index} out of range")
        now = datetime.now(tz=datetime.now().astimezone().tzinfo).isoformat()
        forked = Session(
            id=str(uuid.uuid4()),
            name=name or f"fork of {parent.name or parent.id[:8]} @{fork_index}",
            created_at=now,
            updated_at=now,
            config=parent.config,
            messages=parent.messages[: fork_index + 1],
            compactions=[],
            parent_id=parent.id,
            fork_index=fork_index,
        )
        self._store[forked.id] = forked.to_dict()
        return forked

    def get_fork_points(self, session_id):
        from tau.core.types import ForkInfo
        session = self.load(session_id)
        points = []
        for i, msg in enumerate(session.messages):
            if msg.get("role") == "user":
                preview = (msg.get("content") or "")[:80].replace("\n", " ")
                points.append(ForkInfo(index=i, content=preview))
        return points

    def list_branches(self, session_id):
        return [
            Session.from_dict(d).meta
            for d in self._store.values()
            if d.get("parent_id") == session_id
        ]


# ---------------------------------------------------------------------------
# TauSession — high-level wrapper
# ---------------------------------------------------------------------------


class TauSession:
    """A programmatic session that wraps the Agent for easy embedding.

    Usage::

        session = create_session(provider="openai", model="gpt-4o")
        for event in session.prompt("What files are in the current dir?"):
            print(event.to_dict())
        session.close()

    Or as a context manager::

        with create_session() as session:
            events = list(session.prompt("Hello"))
    """

    def __init__(
        self,
        agent: Agent,
        session: Session,
        session_manager: SessionManager,
        ext_registry: ExtensionRegistry,
        steering: SteeringChannel,
    ) -> None:
        self._agent = agent
        self._session = session
        self._session_manager = session_manager
        self._ext_registry = ext_registry
        self._steering = steering
        self._closed = False

    # -- public API --------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session.id

    @property
    def session(self) -> Session:
        return self._session

    @property
    def agent(self) -> Agent:
        return self._agent

    def prompt(
        self,
        text: str,
        images: list[str] | None = None,
    ) -> Generator[Event, None, None]:
        """Send a message and yield streamed events."""
        if self._closed:
            raise RuntimeError("Session is closed")
        for event in self._agent.run(text, images=images):
            self._ext_registry.fire_hooks(event)
            yield event

    def prompt_sync(
        self,
        text: str,
        images: list[str] | None = None,
    ) -> list[Event]:
        """Send a message and return all events as a list (non-streaming)."""
        return list(self.prompt(text, images=images))

    def steer(self, text: str) -> None:
        """Inject a mid-stream steering message."""
        self._steering.steer(text)

    def enqueue(self, text: str) -> None:
        """Queue a follow-up message."""
        self._steering.enqueue(text)

    def close(self) -> None:
        """Mark the session as closed."""
        self._closed = True

    def __enter__(self) -> "TauSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_session(
    *,
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    workspace: str = ".",
    session_name: str | None = None,
    resume_id: str | None = None,
    session_manager: SessionManager | None = None,
    in_memory: bool = False,
    config: AgentConfig | None = None,
    tau_config: TauConfig | None = None,
    load_skills: bool = True,
    load_extensions: bool = True,
    load_context_files: bool = True,
    shell_confirm: bool = False,
    allowed_tools: list[str] | None = None,
    max_tool_result_chars: int = 0,
) -> TauSession:
    """Create a ready-to-use TauSession.

    Parameters
    ----------
    provider : str, optional
        LLM provider name (openai, google, ollama).
    model : str, optional
        Model name/ID.
    system_prompt : str, optional
        Override the default system prompt.
    workspace : str
        Workspace root directory.
    session_name : str, optional
        Display name for the session.
    resume_id : str, optional
        Resume an existing session by ID or prefix.
    session_manager : SessionManager, optional
        Custom session manager. If None, uses InMemorySessionManager when
        ``in_memory=True``, otherwise the default disk-backed one.
    in_memory : bool
        Use an in-memory session manager (no disk I/O).
    config : AgentConfig, optional
        Full agent config. Fields from other params override it.
    tau_config : TauConfig, optional
        Full tau config. If None, loads from ``~/.tau/config.toml``.
    load_skills : bool
        Whether to discover and load skills.
    load_extensions : bool
        Whether to discover and load extensions.
    load_context_files : bool
        Whether to load AGENTS.md / SYSTEM.md context files.
    shell_confirm : bool
        Whether shell commands require confirmation.

    Returns
    -------
    TauSession
        A session ready for ``prompt()`` calls.
    """
    workspace = str(Path(workspace).resolve())

    # Config
    if tau_config is None:
        ensure_tau_home()
        tau_config = load_config()

    if config is None:
        config = AgentConfig()

    if provider is not None:
        config.provider = provider
    if model is not None:
        config.model = model
    if system_prompt is not None:
        config.system_prompt = system_prompt
    config.workspace_root = workspace

    # Shell / tool config
    tau_config.shell.require_confirmation = shell_confirm

    # Session manager
    if session_manager is None:
        session_manager = InMemorySessionManager() if in_memory else SessionManager()

    steering = SteeringChannel()

    # Tool registry
    registry = ToolRegistry(max_result_chars=max_tool_result_chars)
    register_builtin_tools(registry)
    if allowed_tools is not None:
        registry.keep_only(allowed_tools)
    configure_shell(
        require_confirmation=tau_config.shell.require_confirmation,
        timeout=tau_config.shell.timeout,
        allowed_commands=tau_config.shell.allowed_commands,
        use_persistent_shell=tau_config.shell.use_persistent_shell,
    )
    configure_fs(workspace_root=config.workspace_root)
    configure_context(
        ollama_base_url=tau_config.ollama.base_url,
        ollama_model=config.model,
    )

    # Context
    context = ContextManager(config)

    if load_context_files:
        from tau.context_files import load_system_prompt_override, load_context_files as _load_ctx
        sys_override = load_system_prompt_override(workspace)
        if sys_override is not None:
            config.system_prompt = sys_override
            for m in context._messages:
                if m.role == "system":
                    m.content = sys_override
                    break
        ctx_text = _load_ctx(workspace)
        if ctx_text:
            context.inject_prompt_fragment(ctx_text)

    # Skills
    if load_skills:
        loader = SkillLoader(
            extra_paths=tau_config.skills.paths,
            disabled=tau_config.skills.disabled,
        )
        loader.load_into(registry, context)

    # Extensions
    ext_registry = ExtensionRegistry(
        extra_paths=tau_config.extensions.paths if load_extensions else [],
        disabled=tau_config.extensions.disabled if load_extensions else [],
    )
    if load_extensions:
        ext_registry.load_all(
            registry=registry,
            context=context,
            steering=steering,
            console_print=lambda msg: logger.debug("ext: %s", msg),
            agent_config=config,
        )

    # Resume or create session
    if resume_id:
        session = session_manager.load(resume_id)
        context.restore(session.messages)
    else:
        session = session_manager.new_session(config, name=session_name)

    # Provider
    prov = get_provider(tau_config, config)

    # Agent
    agent = Agent(
        config=config,
        provider=prov,
        registry=registry,
        context=context,
        session=session,
        session_manager=session_manager,
        steering=steering,
    )

    return TauSession(
        agent=agent,
        session=session,
        session_manager=session_manager,
        ext_registry=ext_registry,
        steering=steering,
    )
