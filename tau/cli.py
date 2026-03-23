"""tau CLI — entry point, REPL, and single-shot mode."""
from __future__ import annotations
import json
import logging
import re
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
    click.option("--think", "-t", default=None, type=click.Choice(["off", "minimal", "low", "medium", "high", "xhigh"]), help="Reasoning/thinking effort level."),
    click.option("--image", "-i", multiple=True, help="Attach image file(s) to the first prompt."),
    click.option("--session", "-s", "resume_id", default=None, help="Resume session by ID or prefix."),
    click.option("--session-name", default=None, help="Name for the new session."),
    click.option("--no-confirm", is_flag=True, default=False, help="Disable shell confirmation prompts."),
    click.option("--no-parallel", is_flag=True, default=False, help="Disable parallel tool execution."),
    click.option("--persistent-shell", is_flag=True, default=False, help="Use a persistent bash session across turns."),
    click.option("--workspace", "-w", default=".", show_default=True, help="Workspace root path."),
    click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging and show model thinking tokens."),
    click.option("--mode", "output_mode", type=click.Choice(["interactive", "print", "json", "rpc"]), default=None, help="Output mode: interactive (REPL), print (text only), json (JSONL events), rpc (JSONL over stdio)."),
    click.option("--print", "-P", "print_mode", is_flag=True, default=False, help="Shorthand for --mode print."),
    click.option("--template", "-T", "template_name", default=None, help="Use a prompt template by name (from .tau/prompts/ or ~/.tau/prompts/)."),
    click.option("--var", multiple=True, help="Set template variable: --var key=value (repeatable)."),
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
        use_persistent_shell=tau_config.shell.use_persistent_shell,
        confirm_hook=confirm_hook,
    )
    configure_fs(workspace_root=agent_config.workspace_root)
    configure_context(
        ollama_base_url=tau_config.ollama.base_url,
        ollama_model=agent_config.model,
    )
    context = ContextManager(agent_config)
    # Per-project system prompt override: .tau/SYSTEM.md
    from tau.context_files import load_system_prompt_override
    system_override = load_system_prompt_override(agent_config.workspace_root)
    if system_override is not None:
        agent_config.system_prompt = system_override
        # Replace the system message already created by ContextManager
        for m in context._messages:
            if m.role == "system":
                m.content = system_override
                break
    # Load AGENTS.md / CLAUDE.md context files
    from tau.context_files import load_context_files
    context_text = load_context_files(agent_config.workspace_root)
    if context_text:
        context.inject_prompt_fragment(context_text)
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
    think: str | None,
    no_confirm: bool,
    workspace: str,
    no_parallel: bool = False,
    persistent_shell: bool = False,
) -> AgentConfig:
    if provider:
        tau_config.provider = provider
    if no_confirm:
        tau_config.shell.require_confirmation = False
    if persistent_shell:
        tau_config.shell.use_persistent_shell = True
    return AgentConfig(
        provider=tau_config.provider,
        model=model or tau_config.model,
        thinking_level=think or "off",
        thinking_budgets=tau_config.thinking_budgets.model_dump(),
        max_tokens=tau_config.max_tokens,
        max_turns=tau_config.max_turns,
        system_prompt=tau_config.system_prompt,
        trim_strategy=tau_config.trim_strategy,
        workspace_root=str(Path(workspace).resolve()),
        parallel_tools=tau_config.parallel_tools and not no_parallel,
        parallel_tools_max_workers=tau_config.parallel_tools_max_workers,
    )

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def _render_events(
    agent: Agent,
    user_input: str,
    verbose: bool = False,
    images: list[str] | None = None,
    ext_registry: ExtensionRegistry | None = None,
    output_fn: "Callable[[str], None] | None" = None,
) -> None:
    """Render agent events. If output_fn is provided, all output is sent there
    (for TUI mode). Otherwise writes to sys.stdout / console."""
    stream_buffer: list[str] = []
    is_streaming = False

    def _write(text: str) -> None:
        if output_fn:
            output_fn(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _writeln(text: str = "") -> None:
        _write(text + "\n")

    def _flush_stream(end_line: bool = True) -> None:
        nonlocal is_streaming
        if not is_streaming:
            return
        is_streaming = False
        if end_line:
            _writeln()
        stream_buffer.clear()

    def _confirm_with_flush(command: str) -> bool:
        _flush_stream(end_line=True)
        return _default_confirm(command)

    # In TUI mode, the REPL sets its own confirm hook (_tui_confirm)
    # on the shell module *before* the agent thread starts, so we only
    # override with _confirm_with_flush for the non-TUI (plain stdout) path.
    if output_fn is None:
        from tau.tools import shell as _shell_mod
        _shell_mod._confirm_hook = _confirm_with_flush

    for event in agent.run(user_input, images=images):
        if ext_registry is not None:
            ext_registry.fire_hooks(event)
        if isinstance(event, TextDelta):
            if event.is_thinking:
                continue
            if not is_streaming:
                is_streaming = True
            stream_buffer.append(event.text)
            _write(event.text)
        elif isinstance(event, TextChunk):
            _flush_stream(end_line=True)
        elif isinstance(event, ToolCallEvent):
            _flush_stream(end_line=True)
            args_display = ", ".join(
                f"{k}={str(v)[:60]!r}" for k, v in event.call.arguments.items()
            )
            if output_fn:
                _writeln(f"  ▶ {event.call.name} ({args_display})")
            else:
                console.print(Text.assemble(
                    ("  ▶ ", Style(color="yellow", bold=True)),
                    (event.call.name, Style(color="cyan", bold=True)),
                    (f"({args_display})", Style(color="white", dim=True)),
                ))
        elif isinstance(event, ToolResultEvent):
            r = event.result
            icon = "✗" if r.is_error else "✓"
            preview_lines = r.content.splitlines()[:5]
            preview = "\n    ".join(preview_lines)
            if len(r.content.splitlines()) > 5:
                preview += f"\n    … ({len(r.content.splitlines())} lines total)"
            if output_fn:
                _writeln(f"  {icon} result:\n    {preview}")
            else:
                style = "red dim" if r.is_error else "green dim"
                console.print(Panel(
                    Text(preview, style=style),
                    title=Text(f"{icon} result", style=style),
                    border_style=style,
                    padding=(0, 1),
                ))
        elif isinstance(event, TurnComplete):
            _flush_stream(end_line=True)
            u = event.usage
            tau_config = load_config()
            # session.cumulative_usage is a dict in tau.core.session, but calculate_cost expects an object.
            # wait, calculate_cost uses gettattr. But cumulative_usage is a dict!
            # I must construct a TokenUsage object from cumulative_usage, or update calculate_cost to handle dicts.
            # actually, calculate_cost doing `getattr(..., "input_tokens", 0)` will fail if it's a dict! 
            # So I will pass a dummy object, or just calculate it directly if it's easier.
            cu = getattr(agent._session, "cumulative_usage", {})
            class _DummyUsage:
                input_tokens = cu.get("input_tokens", 0)
                output_tokens = cu.get("output_tokens", 0)
                cache_read_tokens = cu.get("cache_read_tokens", 0)
                cache_write_tokens = cu.get("cache_write_tokens", 0)
            
            session_cost = tau_config.calculate_cost(agent._config.model, _DummyUsage())
            cost_str = f" — session cost: ${session_cost:.3f}" if session_cost > 0 else ""

            if output_fn:
                _writeln(f"  ↳ tokens: {u.input_tokens} in / {u.output_tokens} out{cost_str}")
            else:
                console.print(f"[dim]  ↳ tokens: {u.input_tokens} in / {u.output_tokens} out{cost_str}[/dim]")
        elif isinstance(event, CompactionEvent):
            _flush_stream(end_line=True)
            if event.stage == "start":
                _writeln(f"  ⟳ compacting context ({event.tokens_before:,} tokens)")
            else:
                if event.error:
                    _writeln(f"  ⚠ compaction failed: {event.error}")
                else:
                    saved = event.tokens_before - event.tokens_after
                    _writeln(f"  ✓ compacted {event.tokens_before:,} → {event.tokens_after:,} tokens (saved {saved:,})")
        elif isinstance(event, RetryEvent):
            _flush_stream(end_line=True)
            _writeln(f"  ↻ retrying (attempt {event.attempt}/{event.max_attempts}) in {event.delay:.1f}s — {event.error[:80]}")
        elif isinstance(event, SteerEvent):
            _flush_stream(end_line=True)
            _writeln(f"  ⇢ steered ↳ {event.new_input[:80]}")
            _writeln("─" * 60)
        elif isinstance(event, ExtensionLoadError):
            _flush_stream(end_line=True)
            _writeln(f"  ⚠ extension {event.extension_name!r} failed: {event.error}")
        elif isinstance(event, ErrorEvent):
            _flush_stream(end_line=False)
            _writeln(f"Error: {event.message}")
    _flush_stream(end_line=True)

# ---------------------------------------------------------------------------
# Non-interactive renderers
# ---------------------------------------------------------------------------
def _render_events_print(
    agent: Agent,
    user_input: str,
    images: list[str] | None = None,
    ext_registry: ExtensionRegistry | None = None,
) -> None:
    """Print mode: collect only assistant text, output it to stdout, then exit."""
    text_parts: list[str] = []
    exit_code = 0
    for event in agent.run(user_input, images=images):
        if ext_registry is not None:
            ext_registry.fire_hooks(event)
        if isinstance(event, TextDelta):
            if not event.is_thinking:
                text_parts.append(event.text)
        elif isinstance(event, TextChunk):
            text_parts.append(event.text)
        elif isinstance(event, ErrorEvent):
            print(event.message, file=sys.stderr)
            exit_code = 1
    output = "".join(text_parts)
    if output:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    if exit_code:
        sys.exit(exit_code)


def _render_events_json(
    agent: Agent,
    user_input: str,
    images: list[str] | None = None,
    ext_registry: ExtensionRegistry | None = None,
) -> None:
    """JSON mode: emit each event as a JSON line to stdout (JSONL)."""
    exit_code = 0
    for event in agent.run(user_input, images=images):
        if ext_registry is not None:
            ext_registry.fire_hooks(event)
        if isinstance(event, ErrorEvent):
            exit_code = 1
        sys.stdout.write(json.dumps(event.to_dict(), ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
    if exit_code:
        sys.exit(exit_code)


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
    "  /think <level> hot-swap the reasoning effort (off, low, medium, high)\n"
    "  /tokens        show current token usage\n"
    "  /tree          browse & branch the session history in-place\n"
    "  /prompt <name> [k=v …]  expand a prompt template and send it\n"
    "  /prompts       list available prompt templates\n"
    "  /help          show this message\n"
    "  exit / Ctrl-D  quit"
)

# ---------------------------------------------------------------------------
# /tree — in-place session branch navigator
# ---------------------------------------------------------------------------

# Role icons and colours used by the tree rows
_TREE_ROLE_ICON: dict[str, str] = {
    "user":      "▶",
    "assistant": "◀",
    "tool":      "⚙",
    "system":    "⚑",
}
_TREE_ROLE_COLOUR: dict[str, str] = {
    "user":      "\033[36m",   # cyan
    "assistant": "\033[32m",   # green
    "tool":      "\033[33m",   # yellow
    "system":    "\033[2m",    # dim
}
_TREE_BOLD   = "\033[1m"
_TREE_DIM    = "\033[2m"
_TREE_RESET  = "\033[0m"
_TREE_INVERT = "\033[7m"       # selected-row highlight
_TREE_CYAN   = "\033[36m"
_TREE_RED    = "\033[31m"
_TREE_GREEN  = "\033[32m"


def _tree_navigator(agent: "Agent", output_fn: "Callable[[str], None]") -> None:
    """
    Full-screen, keyboard-driven branch navigator.

    Renders a scrollable list of every message in the current session.
    Keys
    ────
      ↑ / k      move cursor up
      ↓ / j      move cursor down
      Enter       branch: restore context to the selected message index
                  (forks to a new saved session, then replaces the live context)
      f           fork only: save a new session file at the selected point
                  without touching the live context
      Escape / q  cancel and return to the REPL

    The navigator runs as a *blocking* prompt_toolkit Application that takes
    over the full screen, just like the REPL itself.  On exit it hands control
    back to the surrounding REPL application.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import ANSI

    messages = agent._session.messages
    if not messages:
        output_fn("  [dim]  /tree  — no messages in session yet.[/dim]\n")
        return

    total = len(messages)
    cursor: list[int] = [total - 1]   # start at the bottom (most recent)
    status: list[str] = [""]          # one-line status shown at bottom

    # ── helpers ──────────────────────────────────────────────────────────

    def _row(idx: int, selected: bool) -> str:
        msg    = messages[idx]
        role   = msg.get("role", "?")
        icon   = _TREE_ROLE_ICON.get(role, "?")
        colour = _TREE_ROLE_COLOUR.get(role, "")
        # content preview: first non-empty line, max 72 chars
        raw = msg.get("content") or ""
        if isinstance(raw, list):          # multipart (e.g. vision)
            raw = " ".join(
                p.get("text", "") for p in raw if isinstance(p, dict)
            )
        preview = raw.replace("\n", " ").strip()[:72]
        # tool-call badge
        badge = ""
        if msg.get("tool_calls"):
            n = len(msg["tool_calls"])
            badge = f" {_TREE_DIM}[{n} tool{'s' if n != 1 else ''}]{_TREE_RESET}"
        if msg.get("tool_call_id"):
            badge = f" {_TREE_DIM}[tool result]{_TREE_RESET}"

        index_str = f"{idx:3}"
        line = (
            f"  {_TREE_DIM}{index_str}{_TREE_RESET} "
            f"{colour}{icon} {preview}{_TREE_RESET}"
            f"{badge}"
        )
        if selected:
            # Strip existing ANSI before inverting so colours don't bleed
            line = f"{_TREE_INVERT}{_TREE_BOLD}  {index_str} {icon} {preview}{_TREE_RESET}"
        return line + "\n"

    def _header() -> str:
        n = len(messages)
        return (
            f"{_TREE_BOLD}{_TREE_CYAN}  ⎇  session tree{_TREE_RESET}"
            f"  {_TREE_DIM}{n} message{'s' if n != 1 else ''}"
            f"  ·  ↑↓ navigate  ·  Enter branch  ·  f fork-only  ·  Esc cancel"
            f"{_TREE_RESET}\n"
            f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
        )

    def _footer() -> str:
        s = status[0]
        if not s:
            idx = cursor[0]
            msg = messages[idx]
            role = msg.get("role", "?")
            raw = (msg.get("content") or "").replace("\n", " ").strip()
            return (
                f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
                f"  {_TREE_DIM}[{idx}] {role}  {raw[:80]}{_TREE_RESET}\n"
            )
        return (
            f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
            f"  {s}\n"
        )

    def _build_text() -> ANSI:
        parts = [_header()]
        for i in range(total):
            parts.append(_row(i, i == cursor[0]))
        parts.append(_footer())
        return ANSI("".join(parts))

    # ── key bindings ─────────────────────────────────────────────────────

    kb    = KeyBindings()
    result: list[str | None] = [None]   # "branch" | "fork" | None (cancel)
    app_ref: list[Application | None] = [None]

    def _move(delta: int) -> None:
        cursor[0] = max(0, min(total - 1, cursor[0] + delta))
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("up")
    @kb.add("k")
    def _up(_: object) -> None:
        _move(-1)

    @kb.add("down")
    @kb.add("j")
    def _down(_: object) -> None:
        _move(1)

    @kb.add("pageup")
    def _pgup(_: object) -> None:
        _move(-10)

    @kb.add("pagedown")
    def _pgdn(_: object) -> None:
        _move(10)

    @kb.add("home")
    @kb.add("g")
    def _home(_: object) -> None:
        cursor[0] = 0
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("end")
    @kb.add("G")
    def _end(_: object) -> None:
        cursor[0] = total - 1
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("enter")
    def _branch(_: object) -> None:
        result[0] = "branch"
        if app_ref[0]:
            app_ref[0].exit()

    @kb.add("f")
    def _fork_only(_: object) -> None:
        result[0] = "fork"
        if app_ref[0]:
            app_ref[0].exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(_: object) -> None:
        result[0] = None
        if app_ref[0]:
            app_ref[0].exit()

    # ── layout ───────────────────────────────────────────────────────────

    layout = Layout(
        HSplit([
            Window(
                content=FormattedTextControl(_build_text, focusable=True),
                wrap_lines=False,
            ),
        ])
    )

    app: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=True,
        mouse_support=False,
    )
    app_ref[0] = app

    # ── run ──────────────────────────────────────────────────────────────
    # _tree_navigator is called from a *synchronous* key-binding handler
    # (prompt_toolkit's _call_handler never awaits its handlers).
    # The outer REPL Application owns the running asyncio event loop on the
    # main thread, so we cannot call asyncio.run() or loop.run_until_complete()
    # here — both raise "cannot be called from a running event loop".
    #
    # Solution: spin up a dedicated daemon thread with its own fresh event
    # loop, run the tree Application there, and block the main thread with a
    # threading.Event until the tree app exits.  The tree app renders to the
    # same terminal fd — prompt_toolkit handles that correctly as long as only
    # one Application is drawing at a time (the outer REPL is blocked/idle
    # while we wait on the Event).
    import threading as _threading

    _done = _threading.Event()
    _exc: list[BaseException] = []

    def _run_in_thread() -> None:
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(app.run_async())
        except Exception as e:  # noqa: BLE001
            _exc.append(e)
        finally:
            loop.close()
            _done.set()

    t = _threading.Thread(target=_run_in_thread, daemon=True)
    t.start()
    _done.wait()   # block the main thread (key-binding handler) until tree exits

    if _exc:
        output_fn(f"  \033[31m✗ /tree error: {_exc[0]}\033[0m\n")
        return

    # ── act on the user's choice ─────────────────────────────────────────

    chosen_idx = cursor[0]
    action     = result[0]

    if action is None:
        output_fn(f"  {_TREE_DIM}/tree cancelled{_TREE_RESET}\n")
        return

    # Always persist a fork so the old history is safe
    sm     = agent._session_manager
    forked = sm.fork(agent._session.id, chosen_idx)

    if action == "fork":
        output_fn(
            f"  {_TREE_GREEN}✓ forked{_TREE_RESET}"
            f"  session {_TREE_CYAN}{forked.id[:8]}{_TREE_RESET}"
            f"  {_TREE_DIM}({len(forked.messages)} msgs — live context unchanged){_TREE_RESET}\n"
        )
        return

    # action == "branch": replace live context with the snapshot
    snapshot = agent._session.snapshot_at(chosen_idx)
    agent._context.restore(snapshot)

    # Reset compactor overflow flag so the new (smaller) context is clean
    agent._context.compactor.reset_overflow_flag()

    # Update the live session to point at the fork
    agent._session = forked

    output_fn(
        f"  {_TREE_GREEN}⎇  branched{_TREE_RESET}"
        f"  {_TREE_DIM}context rolled back to message [{chosen_idx}]"
        f"  —  old history saved as {forked.id[:8]}{_TREE_RESET}\n"
    )


# ---------------------------------------------------------------------------
# Slash-command dispatcher
# ---------------------------------------------------------------------------

def _handle_slash(
    cmd: str,
    steering: SteeringChannel,
    ext_registry: ExtensionRegistry | None,
    ext_context: ExtensionContext | None,
    agent: "Agent | None" = None,
    output_fn: Callable[[str], None] | None = None,
    staged_images: list[str] | None = None,
) -> bool:
    from typing import Any

    def _print(renderable: Any) -> None:
        if output_fn:
            import io
            f = io.StringIO()
            c = Console(file=f, force_terminal=True, color_system="truecolor")
            c.print(renderable)
            output_fn(f.getvalue())
        else:
            console.print(renderable)

    parts   = cmd.split(None, 1)
    keyword = parts[0].lower()
    arg     = parts[1].strip() if len(parts) > 1 else ""

    if keyword == "/help":
        help_text = _SLASH_HELP
        if ext_registry is not None:
            ext_cmds = ext_registry.all_slash_commands()
            if ext_cmds:
                lines = "\n".join(f"  /{name:<14} {desc}" for name, desc in ext_cmds)
                help_text += f"\n\nExtension commands:\n{lines}"
        _print(Panel(help_text, title="tau slash commands", border_style="dim"))
        return True

    if keyword == "/queue":
        if arg:
            steering.enqueue(arg)
            size = steering.queue_size()
            _print(Text.assemble(
                ("  + queued  ", Style(color="cyan")),
                (f'"{arg[:60]}"', Style(color="cyan", bold=True)),
                (f"  ({size} in queue)", Style(dim=True)),
            ))
        else:
            _print(f"[dim]  Queue size: {steering.queue_size()}[/dim]")
        return True

    if keyword == "/steer":
        if arg:
            steering.steer(arg)
            _print(Text.assemble(
                ("  ⇢ steer set  ", Style(color="magenta")),
                (f'"{arg[:60]}"', Style(color="magenta", bold=True)),
            ))
        else:
            _print("[dim]  Usage: /steer <message>[/dim]")
        return True

    if keyword == "/clear":
        if agent is not None:
            agent._context.restore([])
            _print(Text.assemble(
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
        _print(Text.assemble(
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
            _print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("context compacted", Style(color="green")),
                (f"  {tokens_before:,} → {tokens_after:,} tokens", Style(dim=True)),
                (f"  (saved {saved:,})", Style(color="green", dim=True)),
            ))
        except ValueError as exc:
            _print(f"[yellow dim]  ⚠ cannot compact: {exc}[/yellow dim]")
        except Exception as exc:
            _print(f"[red]  ✗ compaction failed: {exc}[/red]")
        return True

    if keyword == "/model":
        if not arg:
            if agent is not None:
                _print(f"[dim]  Current model: [bold]{agent._config.model}[/bold][/dim]")
                _print("[dim]  Usage: /model <name>[/dim]")
            return True
        if agent is not None:
            old_model = agent._config.model
            agent._config.model = arg
            agent._provider = _swap_provider(agent._config)
            _print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("model swapped", Style(color="green")),
                (f"  {old_model}", Style(dim=True)),
                ("  →  ", Style(dim=True)),
                (arg, Style(color="cyan", bold=True)),
            ))
        return True

    if keyword == "/think":
        levels = {"off", "minimal", "low", "medium", "high", "xhigh"}
        if not arg or arg not in levels:
            if agent is not None:
                _print(f"[dim]  Current thinking level: [bold]{agent._config.thinking_level}[/bold][/dim]")
                _print(f"[dim]  Usage: /think <{'|'.join(levels)}>[/dim]")
            return True
        if agent is not None:
            old_level = agent._config.thinking_level
            agent._config.thinking_level = arg
            _print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("thinking level swapped", Style(color="green")),
                (f"  {old_level}", Style(dim=True)),
                ("  →  ", Style(dim=True)),
                (arg, Style(color="cyan", bold=True)),
            ))
        return True

    if keyword == "/image":
        if not arg:
            _print("[dim]  Usage: /image <path/to/image.png>[/dim]")
            return True
        img_path = Path(arg).resolve()
        if img_path.exists():
            if staged_images is not None:
                staged_images.append(str(img_path))
            _print(f"  [green]✓[/green] staged image: {img_path.name}")
        else:
            _print(f"  [red]✗[/red] image not found: {img_path}")
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
            
            tau_config = load_config()
            cu = getattr(agent._session, "cumulative_usage", {})
            class _DummyUsage:
                input_tokens = cu.get("input_tokens", 0)
                output_tokens = cu.get("output_tokens", 0)
                cache_read_tokens = cu.get("cache_read_tokens", 0)
                cache_write_tokens = cu.get("cache_write_tokens", 0)
            
            session_cost = tau_config.calculate_cost(agent._config.model, _DummyUsage())
            cost_str = f"  (${session_cost:.3f})" if session_cost > 0 else ""

            _print(Text.assemble(
                ("  tokens  ", Style(dim=True)),
                (f"[{bar}]", Style(color=colour)),
                (f"  {used:,} / {budget:,}", Style(color=colour, bold=True)),
                (f"  ({pct}%){cost_str}", Style(dim=True)),
            ))
        return True

    if keyword == "/tree":
        if agent is None:
            _print("[dim]  /tree requires an active session.[/dim]")
            return True
        if output_fn is None:
            _print("[dim]  /tree is only available inside the interactive REPL.[/dim]")
            return True
        _tree_navigator(agent, output_fn)
        return True

    if keyword == "/prompts":
        from tau.prompts import list_templates
        ws = agent._config.workspace_root if agent else "."
        templates = list_templates(ws)
        if not templates:
            _print("[dim]  No prompt templates found.  Add .md files to .tau/prompts/ or ~/.tau/prompts/.[/dim]")
        else:
            from tau.prompts import extract_variables
            lines = []
            for name, path in sorted(templates.items()):
                try:
                    text = path.read_text(encoding="utf-8")
                    vars_ = extract_variables(text)
                    var_str = ", ".join(f"{{{{{v}}}}}" for v in vars_) if vars_ else "(no variables)"
                    loc = "project" if ".tau" in str(path) else "global"
                    lines.append(f"  [cyan]{name:<20}[/cyan] [dim]{var_str:<40} {loc}[/dim]")
                except Exception:  # noqa: BLE001
                    lines.append(f"  [cyan]{name:<20}[/cyan] [red]error reading[/red]")
            _print("[bold]Prompt templates:[/bold]\n" + "\n".join(lines))
        return True

    if keyword == "/prompt":
        if not arg:
            _print("[dim]  Usage: /prompt <template-name> [key=value …][/dim]")
            return True
        if agent is None:
            _print("[dim]  /prompt requires an active session.[/dim]")
            return True
        # Parse: /prompt <name> [key=value key=value ...]
        prompt_parts = arg.split()
        tmpl_name = prompt_parts[0]
        var_pairs = prompt_parts[1:]
        from tau.prompts import resolve_template, parse_var_args
        try:
            variables = parse_var_args(var_pairs)
        except ValueError as e:
            _print(f"[red]  ✗ {e}[/red]")
            return True
        ws = agent._config.workspace_root
        result = resolve_template(tmpl_name, ws, variables)
        if result is None:
            _print(f"[red]  ✗ Template {tmpl_name!r} not found.[/red]")
            return True
        # Return False with a special attribute so the caller sends it as a prompt
        # Actually, we need to dispatch this differently — return the expanded text
        # We'll inject it via the output_fn flow
        _print(f"[dim]  ↳ expanded template [bold]{tmpl_name}[/bold][/dim]")
        return result  # Return the expanded prompt string (truthy, but not True)

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
    staged_images: list[str] | None = None,
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

    import threading
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.layout import Layout, HSplit, Window, ScrollablePane, Float, FloatContainer
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.layout.processors import BeforeInput
    from prompt_toolkit.data_structures import Point

    from tau.editor import (
        expand_at_files,
        complete_path,
        complete_slash_commands,
        is_shell_command,
        run_inline_shell,
        is_image_path,
        detect_pasted_image_macos,
        BUILTIN_SLASH_COMMANDS,
    )

    # -- completer -----------------------------------------------------------
    class _TauCompleter(Completer):
        """Tab completion for /commands and @file references."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            # /slash command completion (at start of line)
            if text.startswith("/"):
                word = text.split()[0] if text.split() else text
                ext_cmds = [name for name, _ in ext_registry.all_slash_commands()] if ext_registry else []
                for cmd in complete_slash_commands(word, BUILTIN_SLASH_COMMANDS, ext_cmds):
                    yield Completion(cmd, start_position=-len(word))
                return
            # @file completion
            # Find the last @token being typed
            m = re.search(r"@([^\s]*)$", text)
            if m:
                prefix = m.group(1)
                ws = agent._config.workspace_root
                for path in complete_path(prefix, ws):
                    yield Completion(path, start_position=-len(prefix))
                return

    # -- state ---------------------------------------------------------------
    agent_running = threading.Event()
    app_ref: list[Application | None] = [None]
    output_text_parts: list[str] = []
    _staged_images = staged_images or []

    # Shell confirmation synchronisation (agent thread ↔ UI thread)
    confirm_pending = threading.Event()      # set while waiting for user answer
    confirm_result: list[bool] = [False]     # shared slot for the answer
    confirm_answered = threading.Event()     # agent thread blocks on this

    # ANSI escape helpers
    _BOLD = "\033[1m"
    _CYAN = "\033[36m"
    _MAGENTA = "\033[35m"
    _DIM = "\033[2m"
    _RESET = "\033[0m"

    header = (
        f"{_BOLD}{_CYAN}tau v{_tau_version()}{_RESET}"
        f"  {_MAGENTA}{agent_config.provider}/{agent_config.model}{_RESET}"
        f"  {_DIM}·  exit or Ctrl-D to quit  ·  /help for commands{_RESET}\n"
        + f"{_DIM}{'═' * 60}{_RESET}\n"
    )
    output_text_parts.append(header)

    # -- buffers -------------------------------------------------------------
    _completer = _TauCompleter()
    input_buffer = Buffer(name="input", completer=_completer, complete_while_typing=False)

    # -- helpers -------------------------------------------------------------
    def _get_output_text() -> ANSI:
        return ANSI("".join(output_text_parts))

    def _get_cursor_position() -> Point:
        # prompt_toolkit splits by \n, so y is exactly the number of \n characters.
        y = sum(text.count('\n') for text in output_text_parts)
        # Set x large enough so that when wrapping, prompt_toolkit keeps the END of the wrapped line visible!
        return Point(x=999999, y=y)

    def _append_output(text: str) -> None:
        output_text_parts.append(text)
        if app_ref[0]:
            app_ref[0].invalidate()

    def _get_prompt_prefix() -> list[tuple[str, str]]:
        if confirm_pending.is_set():
            return [("bold fg:ansiyellow", "allow? [y/N] ")]
        if agent_running.is_set():
            return [("bold fg:ansimagenta", "steer ")]
        return [("bold fg:ansicyan", "you ")]

    # -- agent runner --------------------------------------------------------
    _YELLOW = "\033[33m"

    def _tui_confirm(command: str) -> bool:
        """Confirmation hook for TUI mode — posts the question to the output
        buffer and waits for the user to type y/N in the input field."""
        _append_output(
            f"\n{_BOLD}{_YELLOW}  ⚠  tau wants to run:{_RESET}"
            f"\n\n    {_CYAN}{command}{_RESET}\n\n"
        )
        confirm_answered.clear()
        confirm_pending.set()
        if app_ref[0]:
            app_ref[0].invalidate()
        # Block until the UI thread signals an answer
        confirm_answered.wait()
        return confirm_result[0]

    def _run_agent(user_input: str) -> None:
        agent_running.set()
        if app_ref[0]:
            app_ref[0].invalidate()
        try:
            imgs = list(_staged_images)
            _staged_images.clear()
            _render_events(
                agent, user_input, verbose,
                images=imgs if imgs else None,
                ext_registry=ext_registry,
                output_fn=_append_output,
            )
        except Exception as exc:
            _append_output(f"\nError: {exc}\n")
        finally:
            agent_running.clear()
            if app_ref[0]:
                app_ref[0].invalidate()

    # -- input handler -------------------------------------------------------
    def _on_enter(event: object) -> None:
        text = input_buffer.text.strip()
        input_buffer.reset()

        # If we're waiting for a shell confirmation answer, handle it here
        if confirm_pending.is_set():
            confirm_result[0] = text.lower() in ("y", "yes")
            answer_display = "yes ✓" if confirm_result[0] else "no ✗"
            _append_output(f"  → {answer_display}\n")
            confirm_pending.clear()
            confirm_answered.set()
            return

        if not text:
            return

        if text.lower() in ("exit", "quit", "bye"):
            _append_output("\nGoodbye.\n")
            if app_ref[0]:
                app_ref[0].exit()
            return

        # Inline shell: !command — execute directly, bypass agent
        if is_shell_command(text):
            cmd = text[1:]
            _append_output(f"\n{_DIM}$ {cmd}{_RESET}\n")
            ws = agent._config.workspace_root
            output = run_inline_shell(cmd, ws)
            _append_output(output if output.endswith("\n") else output + "\n")
            return

        # slash commands
        if text.startswith("/") and steering is not None:
            handled = _handle_slash(
                text, steering, ext_registry, ext_context, agent=agent, output_fn=_append_output, staged_images=_staged_images
            )
            if handled is True:
                return
            # /prompt returns the expanded template text as a string
            if isinstance(handled, str):
                text = handled

        # steering (agent already running)
        if agent_running.is_set() and steering is not None:
            steering.steer(text)
            _append_output(f'\n  ⇢ steered "{text[:60]}" (stream will restart)\n')
            return

        # new agent turn — expand @file references first
        ws = agent._config.workspace_root
        expanded, inlined_files = expand_at_files(text, ws)
        if inlined_files:
            n = len(inlined_files)
            names = ", ".join(Path(f).name for f in inlined_files)
            _append_output(f"\n{_BOLD}{_CYAN}you{_RESET} {text}\n")
            _append_output(f"{_DIM}  📎 {n} file{'s' if n > 1 else ''} inlined: {names}{_RESET}\n")
            _append_output(f"{_DIM}{'─' * 60}{_RESET}\n")
            text = expanded
        else:
            _append_output(f"\n{_BOLD}{_CYAN}you{_RESET} {text}\n" + f"{_DIM}{'─' * 60}{_RESET}\n")
        threading.Thread(
            target=_run_agent, args=(text,), daemon=True
        ).start()

    # -- key bindings --------------------------------------------------------
    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: object) -> None:
        _on_enter(event)

    @kb.add("tab")
    def _tab(event: object) -> None:
        """Trigger tab completion or cycle through completions."""
        buff = event.app.current_buffer
        if buff.complete_state:
            # Already showing completions — cycle to next
            buff.complete_next()
        else:
            buff.start_completion(select_first=False)

    @kb.add("c-v")
    def _paste_image(event: object) -> None:
        """Ctrl-V: detect clipboard image (macOS) and stage it."""
        img_path = detect_pasted_image_macos()
        if img_path:
            _staged_images.append(img_path)
            _append_output(f"  {_DIM}📷 image pasted from clipboard → {Path(img_path).name}{_RESET}\n")
        else:
            # Fall back to normal paste
            event.app.clipboard.rotate()
            data = event.app.clipboard.get_data()
            if data.text:
                input_buffer.insert_text(data.text)

    @kb.add("c-d")
    def _ctrl_d(event: object) -> None:
        _append_output("\nGoodbye.\n")
        app_ref[0].exit()  # type: ignore[union-attr]

    @kb.add("c-c")
    def _ctrl_c(event: object) -> None:
        _append_output("\nGoodbye.\n")
        app_ref[0].exit()  # type: ignore[union-attr]

    # -- wire TUI confirm hook into shell module -----------------------------
    from tau.tools import shell as _shell_mod
    _shell_mod._confirm_hook = _tui_confirm

    # -- layout --------------------------------------------------------------
    input_window = Window(
        content=BufferControl(
            buffer=input_buffer,
            input_processors=[BeforeInput(_get_prompt_prefix)],
            focusable=True,
        ),
        height=1,
    )

    body = HSplit([
        Window(
            content=FormattedTextControl(
                _get_output_text,
                focusable=False,
                get_cursor_position=_get_cursor_position
            ),
            wrap_lines=True,
            always_hide_cursor=True,
        ),
        Window(height=1, char="─", style="class:separator"),
        input_window,
    ])

    layout = Layout(
        FloatContainer(
            content=body,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=12, scroll_offset=1),
                ),
            ],
        ),
        focused_element=input_window,
    )

    # -- run -----------------------------------------------------------------
    application: Application[None] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=True,
        mouse_support=True,
    )
    app_ref[0] = application
    application.run()


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
    think: str | None,
    image: tuple[str, ...],
    resume_id: str | None,
    session_name: str | None,
    no_confirm: bool,
    no_parallel: bool,
    persistent_shell: bool,
    workspace: str,
    verbose: bool,
    output_mode: str | None,
    print_mode: bool,
    template_name: str | None,
    var: tuple[str, ...],
) -> None:
    """Run the agent (REPL if no PROMPT given, single-shot otherwise)."""
    # Resolve output mode: --print flag takes precedence as shorthand
    mode = output_mode
    if print_mode:
        mode = "print"

    # Resolve prompt template
    if template_name:
        from tau.prompts import resolve_template, parse_var_args
        try:
            variables = parse_var_args(var)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        ws = str(Path(workspace).resolve())
        rendered = resolve_template(template_name, ws, variables)
        if rendered is None:
            click.echo(f"Error: template {template_name!r} not found.", err=True)
            sys.exit(1)
        # Template becomes the prompt; any PROMPT arg is appended as extra context
        if prompt:
            prompt = rendered.rstrip() + "\n\n" + prompt
        else:
            prompt = rendered

    # Support piped stdin: if no prompt and stdin is not a TTY, read from stdin
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if not prompt:
            click.echo("Error: no prompt provided via argument or stdin.", err=True)
            sys.exit(1)
        # Default to print mode when piped, unless --mode was explicit
        if mode is None:
            mode = "print"

    _setup_logging(verbose)
    ensure_tau_home()
    tau_config = load_config()
    agent_config = _make_agent_config(tau_config, provider, model, think, no_confirm, workspace, no_parallel, persistent_shell)
    session_manager = SessionManager()
    steering = SteeringChannel()

    # Non-interactive modes disable shell confirmation
    confirm_hook = None
    if mode in ("print", "json", "rpc"):
        tau_config.shell.require_confirmation = False

    agent, ext_registry = _build_agent(
        tau_config=tau_config,
        agent_config=agent_config,
        session_manager=session_manager,
        session_name=session_name,
        resume_id=resume_id,
        steering=steering,
    )

    # Expand @file references in the prompt (single-shot mode)
    if prompt:
        from tau.editor import expand_at_files
        prompt, _inlined = expand_at_files(prompt, agent_config.workspace_root)

    if mode == "rpc":
        from tau.rpc import run_rpc
        from tau.sdk import TauSession
        tau_session = TauSession(
            agent=agent,
            session=agent._session,
            session_manager=session_manager,
            ext_registry=ext_registry,
            steering=steering,
        )
        run_rpc(tau_session)
        return

    if mode == "json":
        if not prompt:
            click.echo("Error: --mode json requires a prompt.", err=True)
            sys.exit(1)
        _render_events_json(agent, prompt, images=list(image) if image else None, ext_registry=ext_registry)
    elif mode == "print":
        if not prompt:
            click.echo("Error: --mode print requires a prompt.", err=True)
            sys.exit(1)
        _render_events_print(agent, prompt, images=list(image) if image else None, ext_registry=ext_registry)
    elif prompt:
        _render_events(agent, prompt, verbose, images=list(image) if image else None, ext_registry=ext_registry)
    else:
        _repl(agent, agent_config, verbose, ext_registry=ext_registry, staged_images=list(image) if image else None)

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
    provider: str | None, model: str | None, think: str | None, image: tuple[str, ...], resume_id: str | None,
    session_name: str | None, no_confirm: bool, no_parallel: bool, persistent_shell: bool, workspace: str, verbose: bool,
    output_mode: str | None, print_mode: bool,
    template_name: str | None, var: tuple[str, ...],
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
        agent_config = _make_agent_config(tau_config, provider, model, think, no_confirm, workspace, no_parallel)
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
# `tau prompts` subcommands
# ---------------------------------------------------------------------------
@main.group("prompts")
def prompts_group() -> None:
    """Manage prompt templates."""

@prompts_group.command("list")
@click.option("--workspace", "-w", default=".", show_default=True)
def prompts_list(workspace: str) -> None:
    """List available prompt templates."""
    from tau.prompts import list_templates, extract_variables
    ws = str(Path(workspace).resolve())
    templates = list_templates(ws)
    if not templates:
        console.print("[dim]No prompt templates found.  Add .md files to .tau/prompts/ or ~/.tau/prompts/.[/dim]")
        return
    console.print(f"[bold]{'NAME':<20} {'VARIABLES':<40} {'SOURCE'}[/bold]")
    console.print(Rule(style="dim"))
    for name, path in sorted(templates.items()):
        try:
            text = path.read_text(encoding="utf-8")
            vars_ = extract_variables(text)
            var_str = ", ".join(vars_) if vars_ else "\u2014"
            loc = "project" if ".tau" in str(path) else "global"
            console.print(Text.assemble(
                (f"  {name:<20}", Style(color="cyan", bold=True)),
                (f"{var_str:<40}", Style(dim=True)),
                (loc, Style(dim=True)),
            ))
        except Exception:  # noqa: BLE001
            console.print(f"  [cyan]{name:<20}[/cyan] [red]error reading[/red]")

@prompts_group.command("show")
@click.argument("name")
@click.option("--workspace", "-w", default=".", show_default=True)
def prompts_show(name: str, workspace: str) -> None:
    """Show the contents of a prompt template."""
    from tau.prompts import load_template, extract_variables
    ws = str(Path(workspace).resolve())
    content = load_template(name, ws)
    if content is None:
        err_console.print(f"[red]Template {name!r} not found.[/red]")
        sys.exit(1)
    vars_ = extract_variables(content)
    var_str = ", ".join(f"{{{{{v}}}}}" for v in vars_) if vars_ else "none"
    console.print(Panel(
        content.rstrip(),
        title=f"Template: {name}",
        subtitle=f"variables: {var_str}",
        border_style="cyan",
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

@extensions_group.command("install")
@click.argument("source")
def extensions_install(source: str) -> None:
    """Install an extension package.

    SOURCE is 'git:<url>' or 'pip:<package>'.

    Examples:
      tau extensions install git:https://github.com/user/my-ext
      tau extensions install pip:tau-ext-foobar
    """
    ensure_tau_home()
    from tau.packages import PackageManager, PackageError
    pm = PackageManager()
    try:
        pkg = pm.install(source)
        console.print(Text.assemble(
            ("  ✓ ", Style(color="green", bold=True)),
            ("installed ", Style(color="green")),
            (pkg.name, Style(color="cyan", bold=True)),
            (f"  ({pkg.version})", Style(dim=True)),
        ))
        console.print(f"[dim]    → {pkg.install_path}[/dim]")
        console.print(
            "[yellow dim]  ⚠ Third-party packages run with full system access. "
            "Review source before use.[/yellow dim]"
        )
    except PackageError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@extensions_group.command("remove")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
def extensions_remove(name: str, yes: bool) -> None:
    """Remove an installed extension package by name."""
    ensure_tau_home()
    from tau.packages import PackageManager, PackageNotFoundError
    pm = PackageManager()
    if not yes:
        click.confirm(f"Remove extension package {name!r}?", abort=True)
    try:
        pm.remove(name)
        console.print(Text.assemble(
            ("  ✓ ", Style(color="green", bold=True)),
            ("removed ", Style(color="green")),
            (name, Style(color="cyan", bold=True)),
        ))
    except PackageNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@extensions_group.command("update")
@click.argument("name", required=False)
def extensions_update(name: str | None) -> None:
    """Update installed extension packages.

    If NAME is given, update only that package. Otherwise update all.
    """
    ensure_tau_home()
    from tau.packages import PackageManager, PackageNotFoundError
    pm = PackageManager()
    try:
        updated = pm.update(name)
    except PackageNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    if updated:
        for pkg_name in updated:
            console.print(Text.assemble(
                ("  ✓ ", Style(color="green", bold=True)),
                ("updated ", Style(color="green")),
                (pkg_name, Style(color="cyan", bold=True)),
            ))
    else:
        console.print("[dim]  No packages to update.[/dim]")


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
