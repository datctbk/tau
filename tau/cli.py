"""tau CLI — entry point, REPL, and single-shot mode."""
from __future__ import annotations
import logging
import sys
import threading
from pathlib import Path
from typing import Callable
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
from tau.core.extension import ExtensionContext, ExtensionRegistry
from tau.core.session import SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    CompactionEvent,
    ErrorEvent,
    ExtensionLoadError,
    RetryEvent,
    SteerEvent,
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
from tau.tools.shell import configure_shell, _default_confirm
from tau.core.context import configure_context

console = Console()
err_console = Console(stderr=True)
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
# Shared options
# ---------------------------------------------------------------------------
_AGENT_OPTIONS = [
    click.option("--provider", "-p", default=None, help="LLM provider (openai, google, ollama)."),
    click.option("--model", "-m", default=None, help="Model name."),
    click.option("--session", "-s", "resume_id", default=None, help="Resume session by ID or prefix."),
    click.option("--session-name", default=None, help="Name for the new session."),
    click.option("--no-confirm", is_flag=True, default=False, help="Disable shell confirmation prompts."),
    click.option("--workspace", "-w", default=".", show_default=True, help="Workspace root path."),
    click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging and show model thinking tokens."),
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
    confirm_hook: Callable[[str], bool] | None = None,
    steering: SteeringChannel | None = None,
) -> tuple[Agent, ExtensionRegistry]:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    configure_shell(
        require_confirmation=tau_config.shell.require_confirmation,
        timeout=tau_config.shell.timeout,
        allowed_commands=tau_config.shell.allowed_commands,
        confirm_hook=confirm_hook,
    )
    configure_fs(workspace_root=agent_config.workspace_root)
    configure_context(
        ollama_base_url=tau_config.ollama.base_url,
        ollama_model=agent_config.model,
    )
    context = ContextManager(agent_config)
    loader = SkillLoader(
        extra_paths=tau_config.skills.paths,
        disabled=tau_config.skills.disabled,
    )
    loader.load_into(registry, context)
    ext_registry = ExtensionRegistry(
        extra_paths=tau_config.extensions.paths,
        disabled=tau_config.extensions.disabled,
    )
    ext_registry.load_all(
        registry=registry,
        context=context,
        steering=steering,
        console_print=console.print,
    )
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
    agent = Agent(
        config=agent_config,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=session_manager,
        steering=steering,
    )
    return agent, ext_registry

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
def _render_events(
    agent: Agent,
    user_input: str,
    verbose: bool = False,
    ext_registry: ExtensionRegistry | None = None,
) -> None:
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

    def _confirm_with_flush(command: str) -> bool:
        _flush_stream(end_line=True)
        return _default_confirm(command)

    from tau.tools import shell as _shell_mod
    _shell_mod._confirm_hook = _confirm_with_flush

    for event in agent.run(user_input):
        if ext_registry is not None:
            ext_registry.fire_hooks(event)
        if isinstance(event, TextDelta):
            if event.is_thinking:
                if verbose:
                    sys.stdout.write(f"\x1b[2m{event.text}\x1b[0m")
                    sys.stdout.flush()
                continue
            if not is_streaming:
                is_streaming = True
            stream_buffer.append(event.text)
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, TextChunk):
            _flush_stream(end_line=False)
            console.print(Markdown(event.text))
        elif isinstance(event, ToolCallEvent):
            _flush_stream(end_line=True)
            args_display = ", ".join(
                f"{k}={str(v)[:60]!r}" for k, v in event.call.arguments.items()
            )
            console.print(Text.assemble(
                ("  ▶ ", Style(color="yellow", bold=True)),
                (event.call.name, Style(color="cyan", bold=True)),
                (f"({args_display})", Style(color="white", dim=True)),
            ))
        elif isinstance(event, ToolResultEvent):
            r = event.result
            style = "red dim" if r.is_error else "green dim"
            icon = "✗" if r.is_error else "✓"
            preview_lines = r.content.splitlines()[:5]
            preview = "\n".join(preview_lines)
            if len(r.content.splitlines()) > 5:
                preview += f"\n  … ({len(r.content.splitlines())} lines total)"
            console.print(Panel(
                Text(preview, style=style),
                title=Text(f"{icon} result", style=style),
                border_style=style,
                padding=(0, 1),
            ))
        elif isinstance(event, TurnComplete):
            _flush_stream(end_line=True)
            u = event.usage
            console.print(f"[dim]  ↳ tokens: {u.input_tokens} in / {u.output_tokens} out[/dim]")
        elif isinstance(event, CompactionEvent):
            _flush_stream(end_line=True)
            if event.stage == "start":
                console.print(Text.assemble(
                    ("  ⟳ ", Style(color="yellow", bold=True)),
                    ("compacting context", Style(color="yellow")),
                    (f"  ({event.tokens_before:,} tokens)", Style(dim=True)),
                ))
            else:
                if event.error:
                    console.print(f"[yellow dim]  ⚠ compaction failed: {event.error}[/yellow dim]")
                else:
                    saved = event.tokens_before - event.tokens_after
                    console.print(Text.assemble(
                        ("  ✓ ", Style(color="green", bold=True)),
                        ("context compacted", Style(color="green")),
                        (f"  {event.tokens_before:,} → {event.tokens_after:,} tokens", Style(dim=True)),
                        (f"  (saved {saved:,})", Style(color="green", dim=True)),
                    ))
        elif isinstance(event, RetryEvent):
            _flush_stream(end_line=True)
            console.print(Text.assemble(
                ("  ↻ ", Style(color="yellow", bold=True)),
                (f"retrying (attempt {event.attempt}/{event.max_attempts})", Style(color="yellow")),
                (f"  in {event.delay:.1f}s", Style(dim=True)),
                (f"  — {event.error[:80]}", Style(color="yellow", dim=True)),
            ))
        elif isinstance(event, SteerEvent):
            _flush_stream(end_line=True)
            console.print(Text.assemble(
                ("  ⇢ ", Style(color="magenta", bold=True)),
                ("steered", Style(color="magenta")),
                (f"  ↳ {event.new_input[:80]}", Style(color="magenta", dim=True)),
            ))
            console.print(Rule(style="dim"))
        elif isinstance(event, ExtensionLoadError):
            _flush_stream(end_line=True)
            console.print(f"[yellow dim]  ⚠ extension {event.extension_name!r} failed: {event.error}[/yellow dim]")
        elif isinstance(event, ErrorEvent):
            _flush_stream(end_line=False)
            console.print(f"[bold red]Error:[/bold red] {event.message}")
    _flush_stream(end_line=True)

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
_SLASH_HELP = (
    "  /queue <msg>   add a follow-up prompt to the queue\n"
    "  /queue         show the current queue size\n"
    "  /steer <msg>   send a mid-stream steer\n"
    "  /clear         wipe message history and start fresh\n"
    "  /compact       manually compact the context now\n"
    "  /model <name>  hot-swap the model for this session\n"
    "  /tokens        show current token usage\n"
    "  /help          show this message\n"
    "  exit / Ctrl-D  quit"
)

def _handle_slash(
    cmd: str,
    steering: SteeringChannel,
    ext_registry: ExtensionRegistry | None,
    ext_context: ExtensionContext | None,
    agent: "Agent | None" = None,
) -> bool:
    parts = cmd.split(None, 1)
    keyword = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if keyword == "/help":
        help_text = _SLASH_HELP
        if ext_registry is not None:
            ext_cmds = ext_registry.all_slash_commands()
            if ext_cmds:
                lines = "\n".join(f"  /{name:<14} {desc}" for name, desc in ext_cmds)
                help_text += f"\n\nExtension commands:\n{lines}"
        console.print(Panel(help_text, title="tau slash commands", border_style="dim"))
        return True

    if keyword == "/queue":
        if arg:
            steering.enqueue(arg)
            size = steering.queue_size()
            console.print(Text.assemble(
                ("  + queued  ", Style(color="cyan")),
                (f'"{arg[:60]}"', Style(color="cyan", bold=True)),
                (f"  ({size} in queue)", Style(dim=True)),
            ))
        else:
            console.print(f"[dim]  Queue size: {steering.queue_size()}[/dim]")
        return True

    if keyword == "/steer":
        if arg:
            steering.steer(arg)
            console.print(Text.assemble(
                ("  ⇢ steer set  ", Style(color="magenta")),
                (f'"{arg[:60]}"', Style(color="magenta", bold=True)),
            ))
        else:
            console.print("[dim]  Usage: /steer <message>[/dim]")
        return True

    if keyword == "/clear":
        if agent is not None:
            agent._context.restore([])
            console.print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("context cleared", Style(color="green")),
                ("  — history wiped, system prompt kept", Style(dim=True)),
            ))
        return True

    if keyword == "/compact":
        if agent is None:
            return True
        messages = agent._context.get_messages()
        tokens_before = agent._context.token_count()
        console.print(Text.assemble(
            ("  ⟳ ", Style(color="yellow", bold=True)),
            ("compacting context…", Style(color="yellow")),
            (f"  ({tokens_before:,} tokens)", Style(dim=True)),
        ))
        try:
            new_messages, entry = agent._context.compactor.compact(
                messages, agent._provider, tokens_before
            )
            agent._context.restore([m.to_dict() for m in new_messages if m.role != "system"])
            tokens_after = agent._context.token_count()
            saved = tokens_before - tokens_after
            agent._session_manager.append_compaction(agent._session, entry)
            console.print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("context compacted", Style(color="green")),
                (f"  {tokens_before:,} → {tokens_after:,} tokens", Style(dim=True)),
                (f"  (saved {saved:,})", Style(color="green", dim=True)),
            ))
        except ValueError as exc:
            console.print(f"[yellow dim]  ⚠ cannot compact: {exc}[/yellow dim]")
        except Exception as exc:
            console.print(f"[red]  ✗ compaction failed: {exc}[/red]")
        return True

    if keyword == "/model":
        if not arg:
            if agent is not None:
                console.print(f"[dim]  Current model: [bold]{agent._config.model}[/bold][/dim]")
                console.print("[dim]  Usage: /model <name>[/dim]")
            return True
        if agent is not None:
            old_model = agent._config.model
            agent._config.model = arg
            agent._provider = _swap_provider(agent._config)
            console.print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("model swapped", Style(color="green")),
                (f"  {old_model}", Style(dim=True)),
                ("  →  ", Style(dim=True)),
                (arg, Style(color="cyan", bold=True)),
            ))
        return True

    if keyword == "/tokens":
        if agent is not None:
            used = agent._context.token_count()
            budget = agent._config.max_tokens
            pct = int(used / budget * 100) if budget else 0
            bar_width = 30
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            colour = "green" if pct < 70 else "yellow" if pct < 90 else "red"
            console.print(Text.assemble(
                ("  tokens  ", Style(dim=True)),
                (f"[{bar}]", Style(color=colour)),
                (f"  {used:,} / {budget:,}", Style(color=colour, bold=True)),
                (f"  ({pct}%)", Style(dim=True)),
            ))
        return True

    if ext_registry is not None and ext_context is not None:
        if ext_registry.handle_slash(cmd, ext_context):
            return True
    return False


def _swap_provider(agent_config: "AgentConfig") -> object:
    tau_config = load_config()
    return get_provider(tau_config, agent_config)


def _repl(
    agent: "Agent",
    agent_config: AgentConfig,
    verbose: bool = False,
    ext_registry: ExtensionRegistry | None = None,
) -> None:
    steering: SteeringChannel | None = agent._steering
    ext_context: ExtensionContext | None = None
    if ext_registry is not None and steering is not None:
        ext_context = ExtensionContext(
            registry=agent._registry,
            context=agent._context,
            steering=steering,
            console_print=console.print,
        )

    console.print(Panel(
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
            (" to quit  ·  ", Style(dim=True)),
            ("/help", Style(bold=True)),
            (" for commands", Style(dim=True)),
        ),
        border_style="cyan",
        padding=(0, 1),
    ))

    _agent_running = threading.Event()
    _agent_thread: threading.Thread | None = None

    def _run_agent(user_input: str) -> None:
        _agent_running.set()
        try:
            _render_events(agent, user_input, verbose, ext_registry=ext_registry)
        finally:
            _agent_running.clear()

    def _dispatch(user_input: str) -> None:
        nonlocal _agent_thread
        console.print(Rule(style="dim"))
        _agent_thread = threading.Thread(target=_run_agent, args=(user_input,), daemon=True)
        _agent_thread.start()

    while True:
        if _agent_running.is_set():
            prompt_label = "\n[bold magenta]steer[/bold magenta]"
        else:
            prompt_label = "\n[bold cyan]you[/bold cyan]"
        try:
            user_input = Prompt.ask(prompt_label)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if stripped.startswith("/") and steering is not None:
            if _handle_slash(stripped, steering, ext_registry, ext_context, agent=agent):
                continue

        if _agent_running.is_set() and steering is not None:
            steering.steer(stripped)
            console.print(Text.assemble(
                ("  ⇢ steered  ", Style(color="magenta", bold=True)),
                (f'"{stripped[:60]}"', Style(color="magenta")),
                ("  (stream will restart)", Style(dim=True)),
            ))
            continue

        if _agent_thread is not None:
            _agent_thread.join(timeout=0)
        _dispatch(stripped)


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
    """tau — a minimal CLI coding agent."""

# ---------------------------------------------------------------------------
# `tau run`
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
    steering = SteeringChannel()
    agent, ext_registry = _build_agent(
        tau_config=tau_config,
        agent_config=agent_config,
        session_manager=session_manager,
        session_name=session_name,
        resume_id=resume_id,
        steering=steering,
    )
    if prompt:
        _render_events(agent, prompt, verbose, ext_registry=ext_registry)
    else:
        _repl(agent, agent_config, verbose, ext_registry=ext_registry)

# ---------------------------------------------------------------------------
# `tau sessions` subcommands
# ---------------------------------------------------------------------------
@main.group("sessions")
def sessions_group() -> None:
    """Manage saved sessions."""

@sessions_group.command("list")
def sessions_list() -> None:
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
    sm = SessionManager()
    try:
        session = sm.load(session_id)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    console.print(Panel(
        f"[bold]ID:[/bold]       {session.id}\n"
        f"[bold]Name:[/bold]     {session.name or '—'}\n"
        f"[bold]Provider:[/bold] {session.config.provider}/{session.config.model}\n"
        f"[bold]Created:[/bold]  {session.created_at[:19]}\n"
        f"[bold]Updated:[/bold]  {session.updated_at[:19]}\n"
        f"[bold]Messages:[/bold] {len(session.messages)}",
        title="Session", border_style="cyan",
    ))
    for i, msg in enumerate(session.messages):
        role = msg.get("role", "?")
        content = (msg.get("content") or "")[:200]
        style = {"user": "cyan", "assistant": "green", "tool": "yellow", "system": "dim"}.get(role, "white")
        console.print(f"[{style}][{i}] {role}:[/{style}] {content}")

@sessions_group.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, default=False)
def sessions_delete(session_id: str, yes: bool) -> None:
    sm = SessionManager()
    if not yes:
        click.confirm(f"Delete session {session_id!r}?", abort=True)
    try:
        sm.delete(session_id)
        console.print(f"[green]Deleted session {session_id}.[/green]")
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

@sessions_group.command("fork")
@click.argument("session_id")
@click.argument("message_index", type=int)
@click.option("--name", "-n", default=None)
@click.option("--resume", "-r", is_flag=True, default=False)
@_agent_options
def sessions_fork(
    session_id: str, message_index: int, name: str | None, resume: bool,
    provider: str | None, model: str | None, resume_id: str | None,
    session_name: str | None, no_confirm: bool, workspace: str, verbose: bool,
) -> None:
    sm = SessionManager()
    try:
        forked = sm.fork(session_id, message_index, name=name)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    console.print(Text.assemble(
        ("  ⎇  Forked ", Style(color="green")),
        (session_id[:8], Style(color="cyan", bold=True)),
        (" → ", Style(dim=True)),
        (forked.id[:8], Style(color="cyan", bold=True)),
        (f"  ({len(forked.messages)} messages)", Style(dim=True)),
        (f"  \"{forked.name}\"", Style(color="green", dim=True)),
    ))
    if resume:
        _setup_logging(verbose)
        ensure_tau_home()
        tau_config = load_config()
        agent_config = _make_agent_config(tau_config, provider, model, no_confirm, workspace)
        steering = SteeringChannel()
        agent, ext_reg = _build_agent(
            tau_config=tau_config, agent_config=agent_config,
            session_manager=sm, session_name=None, resume_id=forked.id, steering=steering,
        )
        _repl(agent, agent_config, verbose, ext_registry=ext_reg)

@sessions_group.command("branches")
@click.argument("session_id")
def sessions_branches(session_id: str) -> None:
    sm = SessionManager()
    branches = sm.list_branches(session_id)
    if not branches:
        console.print(f"[dim]No branches found for session {session_id[:8]}.[/dim]")
        return
    console.print(f"[bold]Branches of {session_id[:8]}:[/bold]")
    console.print(Rule(style="dim"))
    for m in branches:
        console.print(m.display())

@sessions_group.command("fork-points")
@click.argument("session_id")
def sessions_fork_points(session_id: str) -> None:
    sm = SessionManager()
    try:
        points = sm.get_fork_points(session_id)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    if not points:
        console.print("[dim]No user messages found.[/dim]")
        return
    console.print(f"[bold]Fork points in {session_id[:8]}:[/bold]")
    console.print(Rule(style="dim"))
    for fp in points:
        console.print(Text.assemble(
            (f"  [{fp.index:3}]  ", Style(color="cyan", bold=True)),
            (fp.content, Style(dim=True)),
        ))

# ---------------------------------------------------------------------------
# `tau extensions` subcommands
# ---------------------------------------------------------------------------
@main.group("extensions")
def extensions_group() -> None:
    """Manage tau extensions."""

@extensions_group.command("list")
def extensions_list() -> None:
    ensure_tau_home()
    tau_config = load_config()
    ext_registry = ExtensionRegistry(
        extra_paths=tau_config.extensions.paths,
        disabled=tau_config.extensions.disabled,
    )
    from tau.core.tool_registry import ToolRegistry as _TR
    from tau.core.context import ContextManager as _CM
    from tau.core.types import AgentConfig as _AC
    ext_registry.load_all(_TR(), _CM(_AC()), steering=None, console_print=lambda _: None)
    manifests = ext_registry.loaded_extensions()
    if not manifests:
        console.print("[dim]No extensions loaded.[/dim]")
        return
    console.print(f"[bold]{'NAME':<20} {'VERSION':<10} {'DESCRIPTION'}[/bold]")
    console.print(Rule(style="dim"))
    for m in manifests:
        author = f"  [dim]by {m.author}[/dim]" if m.author else ""
        console.print(Text.assemble(
            (f"  {m.name:<20}", Style(color="cyan", bold=True)),
            (f"{m.version:<10}", Style(dim=True)),
            (m.description, Style()),
            (author, Style()),
        ))
    slash_cmds = ext_registry.all_slash_commands()
    if slash_cmds:
        console.print()
        console.print("[bold]Slash commands:[/bold]")
        for cmd, desc in slash_cmds:
            console.print(f"  [cyan]/{cmd}[/cyan]  [dim]{desc}[/dim]")

@extensions_group.command("show")
@click.argument("name")
def extensions_show(name: str) -> None:
    ensure_tau_home()
    tau_config = load_config()
    ext_registry = ExtensionRegistry(
        extra_paths=tau_config.extensions.paths,
        disabled=tau_config.extensions.disabled,
    )
    from tau.core.tool_registry import ToolRegistry as _TR
    from tau.core.context import ContextManager as _CM
    from tau.core.types import AgentConfig as _AC
    ext_registry.load_all(_TR(), _CM(_AC()), steering=None, console_print=lambda _: None)
    ext = ext_registry.get(name)
    if ext is None:
        err_console.print(f"[red]Extension {name!r} not found.[/red]")
        sys.exit(1)
    m = ext.manifest
    tools = ext.tools()
    cmds = ext.slash_commands()
    has_hook = type(ext).event_hook is not type(ext).__mro__[1].event_hook  # type: ignore[index]
    console.print(Panel(
        f"[bold]Name:[/bold]        {m.name}\n"
        f"[bold]Version:[/bold]     {m.version}\n"
        f"[bold]Author:[/bold]      {m.author or '—'}\n"
        f"[bold]Description:[/bold] {m.description or '—'}\n"
        f"[bold]Tools:[/bold]       {', '.join(t.name for t in tools) or '—'}\n"
        f"[bold]Slash cmds:[/bold]  {', '.join('/' + c.name for c in cmds) or '—'}\n"
        f"[bold]Event hook:[/bold]  {'yes' if has_hook else 'no'}\n"
        + (f"[bold]Prompt frag:[/bold] yes\n" if m.system_prompt_fragment else ""),
        title=f"Extension: {m.name}", border_style="cyan",
    ))

# ---------------------------------------------------------------------------
# `tau config` subcommands
# ---------------------------------------------------------------------------
@main.group("config")
def config_group() -> None:
    """View or update configuration."""

@config_group.command("show")
def config_show() -> None:
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
        title="tau config", border_style="cyan",
    ))

_VALID_CONFIG_KEYS = {"provider", "model", "max_tokens", "max_turns", "trim_strategy"}

@config_group.command("set")
@click.argument("key", type=click.Choice(sorted(_VALID_CONFIG_KEYS), case_sensitive=False))
@click.argument("value")
def config_set(key: str, value: str) -> None:
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
