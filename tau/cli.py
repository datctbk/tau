"""tau CLI — entry point, REPL, and single-shot mode."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

from tau.config import TauConfig, ensure_tau_home, load_config
from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import SessionManager
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    TextChunk,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnComplete,
)
from tau.providers import get_provider
from tau.skills import SkillLoader
from tau.tools import register_builtin_tools
from tau.tools.fs import configure_fs
from tau.tools.shell import configure_shell

console = Console()
err_console = Console(stderr=True)
# A dedicated console for streaming chunks — forces terminal mode so output
# is never suppressed regardless of whether stdout is a TTY.
_stream_console = Console(highlight=False, markup=False, soft_wrap=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Shared options (reused across run + root invoke)
# ---------------------------------------------------------------------------

_AGENT_OPTIONS = [
    click.option("--provider", "-p", default=None, help="LLM provider (openai, google, ollama)."),
    click.option("--model", "-m", default=None, help="Model name."),
    click.option("--session", "-s", "resume_id", default=None, help="Resume session by ID or prefix."),
    click.option("--session-name", default=None, help="Name for the new session."),
    click.option("--no-confirm", is_flag=True, default=False, help="Disable shell confirmation prompts."),
    click.option("--workspace", "-w", default=".", show_default=True, help="Workspace root path."),
    click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."),
]


def _agent_options(fn):
    for option in reversed(_AGENT_OPTIONS):
        fn = option(fn)
    return fn


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(
    tau_config: TauConfig,
    agent_config: AgentConfig,
    session_manager: SessionManager,
    session_name: str | None,
    resume_id: str | None,
) -> Agent:
    registry = ToolRegistry()
    register_builtin_tools(registry)

    configure_shell(
        require_confirmation=tau_config.shell.require_confirmation,
        timeout=tau_config.shell.timeout,
        allowed_commands=tau_config.shell.allowed_commands,
    )
    configure_fs(workspace_root=agent_config.workspace_root)

    context = ContextManager(agent_config)

    loader = SkillLoader(
        extra_paths=tau_config.skills.paths,
        disabled=tau_config.skills.disabled,
    )
    loader.load_into(registry, context)

    if resume_id:
        session = session_manager.load(resume_id)
        context.restore(session.messages)
        console.print(
            f"[dim]Resumed session [bold]{session.id[:8]}[/bold]"
            + (f" ({session.name})" if session.name else "")
            + "[/dim]"
        )
    else:
        session = session_manager.new_session(agent_config, name=session_name)

    provider = get_provider(tau_config, agent_config)

    return Agent(
        config=agent_config,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=session_manager,
    )


def _make_agent_config(
    tau_config: TauConfig,
    provider: str | None,
    model: str | None,
    no_confirm: bool,
    workspace: str,
) -> AgentConfig:
    if provider:
        tau_config.provider = provider
    if no_confirm:
        tau_config.shell.require_confirmation = False
    return AgentConfig(
        provider=tau_config.provider,
        model=model or tau_config.model,
        max_tokens=tau_config.max_tokens,
        max_turns=tau_config.max_turns,
        system_prompt=tau_config.system_prompt,
        trim_strategy=tau_config.trim_strategy,
        workspace_root=str(Path(workspace).resolve()),
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _render_events(agent: Agent, user_input: str) -> None:
    stream_buffer: list[str] = []
    is_streaming = False

    def _flush_stream(end_line: bool = True) -> None:
        nonlocal is_streaming
        if not is_streaming:
            return
        is_streaming = False
        if end_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
        stream_buffer.clear()

    for event in agent.run(user_input):

        if isinstance(event, TextDelta):
            if not is_streaming:
                is_streaming = True
            stream_buffer.append(event.text)
            # Write directly to stdout — no buffering, no TTY detection
            sys.stdout.write(event.text)
            sys.stdout.flush()

        elif isinstance(event, TextChunk):
            # Blocking mode — render directly as Markdown
            _flush_stream(end_line=False)
            console.print(Markdown(event.text))

        elif isinstance(event, ToolCallEvent):
            _flush_stream(end_line=True)
            args_display = ", ".join(
                f"{k}={str(v)[:60]!r}" for k, v in event.call.arguments.items()
            )
            console.print(
                Text.assemble(
                    ("  ▶ ", Style(color="yellow", bold=True)),
                    (event.call.name, Style(color="cyan", bold=True)),
                    (f"({args_display})", Style(color="white", dim=True)),
                )
            )

        elif isinstance(event, ToolResultEvent):
            r = event.result
            style = "red dim" if r.is_error else "green dim"
            icon = "✗" if r.is_error else "✓"
            preview_lines = r.content.splitlines()[:5]
            preview = "\n".join(preview_lines)
            if len(r.content.splitlines()) > 5:
                preview += f"\n  … ({len(r.content.splitlines())} lines total)"
            console.print(
                Panel(
                    Text(preview, style=style),
                    title=Text(f"{icon} result", style=style),
                    border_style=style,
                    padding=(0, 1),
                )
            )

        elif isinstance(event, TurnComplete):
            _flush_stream(end_line=True)
            u = event.usage
            console.print(
                f"[dim]  ↳ tokens: {u.input_tokens} in / {u.output_tokens} out[/dim]"
            )

        elif isinstance(event, ErrorEvent):
            _flush_stream(end_line=False)
            console.print(f"[bold red]Error:[/bold red] {event.message}")

    _flush_stream(end_line=True)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _repl(agent: Agent, agent_config: AgentConfig) -> None:
    console.print(
        Panel(
            Text.assemble(
                ("tau ", Style(bold=True, color="cyan")),
                (f"v{_tau_version()}  ", Style(dim=True)),
                (agent_config.provider, Style(color="magenta")),
                ("/", Style(dim=True)),
                (agent_config.model, Style(color="magenta")),
                ("  ·  ", Style(dim=True)),
                ("exit", Style(bold=True)),
                (" or ", Style(dim=True)),
                ("Ctrl-D", Style(bold=True)),
                (" to quit.", Style(dim=True)),
            ),
            border_style="cyan",
            padding=(0, 1),
        )
    )
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]you[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break
        console.print(Rule(style="dim"))
        _render_events(agent, stripped)


def _tau_version() -> str:
    try:
        from tau import __version__
        return __version__
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """tau — a minimal CLI coding agent.

    \b
    Quick start:
      tau run                        # interactive REPL
      tau run "fix the bug in foo.py"  # single-shot
      tau run -p ollama -m llama3 "explain this code"
    """


# ---------------------------------------------------------------------------
# `tau run` — the main agent command
# ---------------------------------------------------------------------------

@main.command("run")
@click.argument("prompt", required=False)
@_agent_options
def run_cmd(
    prompt: str | None,
    provider: str | None,
    model: str | None,
    resume_id: str | None,
    session_name: str | None,
    no_confirm: bool,
    workspace: str,
    verbose: bool,
) -> None:
    """Run the agent (REPL if no PROMPT given, single-shot otherwise)."""
    _setup_logging(verbose)
    ensure_tau_home()

    tau_config = load_config()
    agent_config = _make_agent_config(tau_config, provider, model, no_confirm, workspace)
    session_manager = SessionManager()

    agent = _build_agent(
        tau_config=tau_config,
        agent_config=agent_config,
        session_manager=session_manager,
        session_name=session_name,
        resume_id=resume_id,
    )

    if prompt:
        _render_events(agent, prompt)
    else:
        _repl(agent, agent_config)


# ---------------------------------------------------------------------------
# `tau sessions` subcommands
# ---------------------------------------------------------------------------

@main.group("sessions")
def sessions_group() -> None:
    """Manage saved sessions."""


@sessions_group.command("list")
def sessions_list() -> None:
    """List all saved sessions."""
    sm = SessionManager()
    metas = sm.list_sessions()
    if not metas:
        console.print("[dim]No sessions found.[/dim]")
        return
    console.print(f"[bold]{'ID':10} {'NAME':24} {'MODEL':22} {'UPDATED':19}[/bold]")
    console.print(Rule(style="dim"))
    for m in metas:
        console.print(m.display())


@sessions_group.command("show")
@click.argument("session_id")
def sessions_show(session_id: str) -> None:
    """Show message history for a session."""
    sm = SessionManager()
    try:
        session = sm.load(session_id)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]ID:[/bold]       {session.id}\n"
            f"[bold]Name:[/bold]     {session.name or '—'}\n"
            f"[bold]Provider:[/bold] {session.config.provider}/{session.config.model}\n"
            f"[bold]Created:[/bold]  {session.created_at[:19]}\n"
            f"[bold]Updated:[/bold]  {session.updated_at[:19]}\n"
            f"[bold]Messages:[/bold] {len(session.messages)}",
            title="Session",
            border_style="cyan",
        )
    )
    for i, msg in enumerate(session.messages):
        role = msg.get("role", "?")
        content = (msg.get("content") or "")[:200]
        style = {
            "user": "cyan", "assistant": "green",
            "tool": "yellow", "system": "dim",
        }.get(role, "white")
        console.print(f"[{style}][{i}] {role}:[/{style}] {content}")


@sessions_group.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
def sessions_delete(session_id: str, yes: bool) -> None:
    """Delete a session."""
    sm = SessionManager()
    if not yes:
        click.confirm(f"Delete session {session_id!r}?", abort=True)
    try:
        sm.delete(session_id)
        console.print(f"[green]Deleted session {session_id}.[/green]")
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# `tau config` subcommands
# ---------------------------------------------------------------------------

@main.group("config")
def config_group() -> None:
    """View or update configuration."""


@config_group.command("show")
def config_show() -> None:
    """Print the current configuration."""
    from tau.config import CONFIG_PATH
    tau_config = load_config()
    console.print(Panel(
        f"[bold]Config file:[/bold] {CONFIG_PATH}\n\n"
        f"[bold]provider:[/bold]      {tau_config.provider}\n"
        f"[bold]model:[/bold]         {tau_config.model}\n"
        f"[bold]max_tokens:[/bold]    {tau_config.max_tokens}\n"
        f"[bold]max_turns:[/bold]     {tau_config.max_turns}\n"
        f"[bold]trim_strategy:[/bold] {tau_config.trim_strategy}\n"
        f"[bold]shell.confirm:[/bold] {tau_config.shell.require_confirmation}\n"
        f"[bold]shell.timeout:[/bold] {tau_config.shell.timeout}s",
        title="tau config",
        border_style="cyan",
    ))


_VALID_CONFIG_KEYS = {"provider", "model", "max_tokens", "max_turns", "trim_strategy"}


@config_group.command("set")
@click.argument("key", type=click.Choice(sorted(_VALID_CONFIG_KEYS), case_sensitive=False))
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config value.  e.g. tau config set model gpt-4o"""
    from tau.config import CONFIG_PATH
    ensure_tau_home()

    existing = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    lines = existing.splitlines()
    in_defaults = False
    key_written = False
    new_lines: list[str] = []
    val_str = f'"{value}"' if not value.lstrip("-").isdigit() else value

    for line in lines:
        stripped = line.strip()
        if stripped == "[defaults]":
            in_defaults = True
            new_lines.append(line)
            continue
        if in_defaults and stripped.startswith("[") and stripped != "[defaults]":
            if not key_written:
                new_lines.append(f'{key} = {val_str}')
                key_written = True
            in_defaults = False
        if in_defaults and (stripped.startswith(f"{key} ") or stripped.startswith(f"{key}=")):
            new_lines.append(f'{key} = {val_str}')
            key_written = True
            continue
        new_lines.append(line)

    if not key_written:
        if not any(l.strip() == "[defaults]" for l in new_lines):
            new_lines.append("")
            new_lines.append("[defaults]")
        new_lines.append(f'{key} = {val_str}')

    CONFIG_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    console.print(f"[green]Set [bold]{key}[/bold] = {value}[/green]")
