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
from tau.config import TauConfig, ensure_tau_home, load_config, get_theme_file_paths
from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.extension import ExtensionContext, ExtensionRegistry
from tau.core.session import SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    CompactionEvent,
    CostLimitExceeded,
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
# Theme — centralised colours loaded from config
# ---------------------------------------------------------------------------
class _Theme:
    """Lazy theme that reads from TauConfig on first access."""
    _loaded: bool = False
    user_color: str = "cyan"
    assistant_color: str = "green"
    tool_color: str = "yellow"
    system_color: str = "dim"
    error_color: str = "red"
    accent_color: str = "cyan"
    success_color: str = "green"
    warning_color: str = "yellow"
    border_style: str = "dim"

    @classmethod
    def load(cls, tau_config: "TauConfig | None" = None, *, force: bool = False) -> None:
        if cls._loaded and not force:
            return
        if tau_config is None:
            try:
                tau_config = load_config()
            except Exception:
                pass
        if tau_config is not None:
            t = tau_config.theme
            cls.user_color = t.user_color
            cls.assistant_color = t.assistant_color
            cls.tool_color = t.tool_color
            cls.system_color = t.system_color
            cls.error_color = t.error_color
            cls.accent_color = t.accent_color
            cls.success_color = t.success_color
            cls.warning_color = t.warning_color
            cls.border_style = t.border_style
        cls._loaded = True

theme = _Theme()


class _ThemeWatcher:
    """Background poller that hot-reloads the active theme when its source
    file changes on disk.

    Watches ``~/.tau/config.toml`` and, if present, ``~/.tau/theme.toml``.
    On any mtime change the config is re-read and ``theme`` is updated in
    place; the prompt_toolkit ``Application`` is then invalidated so the
    new colours take effect on the very next render.
    """

    _INTERVAL = 0.5  # seconds between polls

    def __init__(self, app_ref: "list[Application | None]") -> None:
        self._app_ref = app_ref
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    def _snapshot(self) -> None:
        """Record current mtimes of all watched paths."""
        for p in get_theme_file_paths():
            self._mtimes[str(p)] = self._mtime(p)

    def _check(self) -> None:
        """Poll watched paths; reload theme on any change."""
        changed = False
        for p in get_theme_file_paths():
            key = str(p)
            mtime = self._mtime(p)
            if self._mtimes.get(key, -1.0) != mtime:
                self._mtimes[key] = mtime
                changed = True
        if changed:
            try:
                cfg = load_config()
                theme.load(cfg, force=True)
            except Exception:  # noqa: BLE001
                pass
            app = self._app_ref[0]
            if app is not None:
                try:
                    app.invalidate()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        self._stop.clear()
        self._snapshot()

        def _run() -> None:
            while not self._stop.wait(self._INTERVAL):
                self._check()

        self._thread = threading.Thread(
            target=_run, daemon=True, name="tau-theme-watcher"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the watcher to exit (does not join the thread)."""
        self._stop.set()

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
    click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging."),
    click.option("--show-thinking", is_flag=True, default=False, help="Stream model thinking/reasoning tokens (dim italic)."),
    click.option("--mode", "output_mode", type=click.Choice(["interactive", "print", "json", "rpc"]), default=None, help="Output mode: interactive (REPL), print (text only), json (JSONL events), rpc (JSONL over stdio)."),
    click.option("--print", "-P", "print_mode", is_flag=True, default=False, help="Shorthand for --mode print."),
    click.option("--template", "-T", "template_name", default=None, help="Use a prompt template by name (from .tau/prompts/ or ~/.tau/prompts/)."),
    click.option("--var", multiple=True, help="Set template variable: --var key=value (repeatable)."),
    click.option("--max-cost", "max_cost", type=float, default=None, help="USD budget ceiling; stop session when exceeded."),
    click.option("--no-session", is_flag=True, default=False, help="Ephemeral mode: don't persist the session to disk."),
    click.option("--trace-log", "trace_log", is_flag=False, flag_value="__default__", default=None, help="Log full LLM requests/responses to a file (default: <workspace>/tau-trace.log)."),
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

    # Apply configurable tool set: disabled / enabled_only
    if tau_config.tools.enabled_only:
        allowed = set(tau_config.tools.enabled_only)
        for name in list(registry.names()):
            if name not in allowed:
                registry.unregister(name)
    elif tau_config.tools.disabled:
        for name in tau_config.tools.disabled:
            registry.unregister(name)

    configure_shell(
        require_confirmation=tau_config.shell.require_confirmation,
        timeout=tau_config.shell.timeout,
        allowed_commands=tau_config.shell.allowed_commands,
        use_persistent_shell=tau_config.shell.use_persistent_shell,
        confirm_hook=confirm_hook,
        workspace_root=agent_config.workspace_root,
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
        cost_calculator=tau_config.calculate_cost,
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
    max_cost: float | None = None,
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
        max_cost=max_cost if max_cost is not None else tau_config.max_cost,
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
    show_thinking: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> None:
    """Render agent events. If output_fn is provided, all output is sent there
    (for TUI mode). Otherwise writes to sys.stdout / console."""
    stream_buffer: list[str] = []
    is_streaming = False
    _in_thinking = False  # track whether we're currently in a thinking block

    def _write(text: str) -> None:
        if output_fn:
            output_fn(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _writeln(text: str = "") -> None:
        _write(text + "\n")

    def _flush_stream(end_line: bool = True) -> None:
        nonlocal is_streaming, _in_thinking
        if _in_thinking:
            # Close thinking styling
            if output_fn:
                _write("\033[0m")
            else:
                sys.stdout.write("\033[0m")
                sys.stdout.flush()
            _in_thinking = False
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

    # Spinner for plain stdout mode (non-TUI).
    # Single long-lived thread; pause/resume via _spin_active event to avoid
    # race conditions when starting a new spinner immediately after stopping.
    import threading as _thr
    _SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _spin_active = _thr.Event()   # set = spinning, clear = paused
    _spin_exit = _thr.Event()     # set = thread should terminate
    _spin_msg = ["thinking..."]

    def _spin_loop() -> None:
        idx = 0
        while not _spin_exit.is_set():
            if _spin_active.is_set():
                sys.stdout.write(f"\r  {_SPIN[idx % len(_SPIN)]} {_spin_msg[0]}")
                sys.stdout.flush()
                idx += 1
            _spin_exit.wait(0.08)
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()

    def _start_spinner(msg: str = "thinking...") -> None:
        if output_fn is not None:
            return
        _spin_msg[0] = msg
        _spin_active.set()

    def _stop_spinner() -> None:
        if _spin_active.is_set():
            _spin_active.clear()
            if output_fn is None:
                sys.stdout.write("\r" + " " * 30 + "\r")
                sys.stdout.flush()

    def _kill_spinner() -> None:
        _spin_active.clear()
        _spin_exit.set()

    if output_fn is None:
        _thr.Thread(target=_spin_loop, daemon=True).start()
    _start_spinner("thinking...")

    gen = agent.run(user_input, images=images)
    for event in gen:
        # --- Escape-to-cancel ---
        if cancel_event is not None and cancel_event.is_set():
            _stop_spinner()
            _flush_stream(end_line=True)
            _write("\n⚠ Cancelled.\n")
            if hasattr(gen, "close"):
                gen.close()
            break
        if ext_registry is not None:
            ext_registry.fire_hooks(event)
        if isinstance(event, TextDelta):
            if event.is_thinking:
                if not show_thinking:
                    continue
                _stop_spinner()
                # Start thinking block: dim italic + gray color
                if not _in_thinking:
                    _in_thinking = True
                    # \033[2m = dim, \033[3m = italic, \033[90m = dark gray
                    if output_fn:
                        _write("\033[2;3;90m")
                    else:
                        sys.stdout.write("\033[2;3;90m")
                        sys.stdout.flush()
                if output_fn:
                    _write(event.text)
                else:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                continue
            # Transition from thinking → answer: close styling, add separator
            if _in_thinking:
                if output_fn:
                    _write("\033[0m\n")
                else:
                    sys.stdout.write("\033[0m\n")
                    sys.stdout.flush()
                _in_thinking = False
            _stop_spinner()
            if not is_streaming:
                is_streaming = True
            stream_buffer.append(event.text)
            _write(event.text)
        elif isinstance(event, TextChunk):
            _stop_spinner()
            _flush_stream(end_line=True)
        elif isinstance(event, ToolCallEvent):
            _stop_spinner()
            _flush_stream(end_line=True)
            args_display = ", ".join(
                f"{k}={str(v)[:60]!r}" for k, v in event.call.arguments.items()
            )
            if output_fn:
                _writeln(f"  ▶ {event.call.name} ({args_display})")
            else:
                console.print(Text.assemble(
                    ("  ▶ ", Style(color=theme.tool_color, bold=True)),
                    (event.call.name, Style(color=theme.accent_color, bold=True)),
                    (f"({args_display})", Style(color="white", dim=True)),
                ))
            _start_spinner("running...")
        elif isinstance(event, ToolResultEvent):
            _stop_spinner()
            r = event.result
            icon = "✗" if r.is_error else "✓"
            preview_lines = r.content.splitlines()[:5]
            preview = "\n    ".join(preview_lines)
            if len(r.content.splitlines()) > 5:
                preview += f"\n    … ({len(r.content.splitlines())} lines total)"
            if output_fn:
                _writeln(f"  {icon} result:\n    {preview}")
            else:
                style = f"{theme.error_color} dim" if r.is_error else f"{theme.success_color} dim"
                console.print(Panel(
                    Text(preview, style=style),
                    title=Text(f"{icon} result", style=style),
                    border_style=style,
                    padding=(0, 1),
                ))
            _start_spinner("thinking...")
        elif isinstance(event, TurnComplete):
            _stop_spinner()
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
            _stop_spinner()
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
            _stop_spinner()
            _flush_stream(end_line=True)
            _writeln(f"  ↻ retrying (attempt {event.attempt}/{event.max_attempts}) in {event.delay:.1f}s — {event.error[:80]}")
        elif isinstance(event, SteerEvent):
            _stop_spinner()
            _flush_stream(end_line=True)
            _writeln(f"  ⇢ steered ↳ {event.new_input[:80]}")
            _writeln("─" * 60)
        elif isinstance(event, ExtensionLoadError):
            _stop_spinner()
            _flush_stream(end_line=True)
            _writeln(f"  ⚠ extension {event.extension_name!r} failed: {event.error}")
        elif isinstance(event, ErrorEvent):
            _stop_spinner()
            _flush_stream(end_line=False)
            _writeln(f"Error: {event.message}")
        elif isinstance(event, CostLimitExceeded):
            _stop_spinner()
            _flush_stream(end_line=True)
            if output_fn:
                _writeln(f"  ⚠ budget exceeded: ${event.session_cost:.3f} >= ${event.max_cost:.3f} — stopping session")
            else:
                console.print(f"[bold red]  ⚠ budget exceeded: ${event.session_cost:.3f} >= ${event.max_cost:.3f} — stopping session[/bold red]")
    _kill_spinner()
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
        elif isinstance(event, CostLimitExceeded):
            print(f"Budget exceeded: ${event.session_cost:.3f} >= ${event.max_cost:.3f}", file=sys.stderr)
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
    "  /fork [name]   fork session at current position into a named branch\n"
    "  /bookmark      list bookmarks for current session\n"
    "  /bookmark <i> [label]  toggle bookmark at message index i\n"
    "  /copy          copy last assistant message to clipboard\n"
    "  /export [file]  export session as JSON (or .md for markdown)\n"
    "  /share [--json]  upload session to paste.rs and print the URL\n"
    "  /import <file>  import a previously-exported JSON session file\n"

    "  /reload        hot-reload config, extensions, skills, context files\n"
    "  /prompt <name> [k=v …]  expand a prompt template and send it\n"
    "  /prompts       list available prompt templates\n"
    "  /themes        list available theme presets\n"
    "  /theme <name>  switch to a theme preset (instant, no restart)\n"
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


def _tree_navigator(
    agent: "Agent",
    output_fn: "Callable[[str], None]",
    reset_fn: "Callable[[], None] | None" = None,
) -> None:
    """
    Full-screen session tree navigator with cross-branch navigation and bookmarks.

    Keys
    ────
      ↑ / k        move cursor up
      ↓ / j        move cursor down
      PgUp/PgDn    scroll 10 lines
      g / Home     first message
      G / End      last message
      Enter        branch: fork + restore context to selected message
      f            fork-only: save branch without changing live context
      b            toggle bookmark on selected message
      B            jump to next bookmark
      → / l        navigate into a child branch (if any at selected row)
      ← / h        go back to parent session view
      Esc / q      cancel and return to REPL
    """
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.formatted_text import ANSI

    sm = agent._session_manager

    # ── nav frame helpers ────────────────────────────────────────────────

    def _load_frame(session: "object") -> "dict":
        """Build a navigation frame dict for *session*."""
        branches: "dict[int, list]" = {}
        try:
            for child_meta in sm.list_branches(session.id):
                idx = child_meta.fork_index
                if idx is not None:
                    branches.setdefault(idx, []).append(child_meta)
        except Exception:
            pass
        return {
            "session": session,
            "branches": branches,
            "bookmarked": {b["index"] for b in session.bookmarks},
        }

    if not agent._session.messages:
        output_fn("  \033[2m/tree — no messages in session yet.\033[0m\n")
        return

    _nav_stack: "list[dict]" = [_load_frame(agent._session)]

    cursor: "list[int]" = [len(agent._session.messages) - 1]
    status: "list[str]" = [""]
    # mode: "list" | "pick_child"
    mode: "list[str]" = ["list"]
    pick_children: "list" = []
    pick_cursor: "list[int]" = [0]
    result: "list[str | None]" = [None]   # "branch" | "fork" | None
    app_ref: "list[Application | None]" = [None]

    # ── frame accessors ──────────────────────────────────────────────────

    def _frame() -> "dict":
        return _nav_stack[-1]

    def _cur_session() -> "object":
        return _frame()["session"]

    def _cur_msgs() -> "list":
        return _cur_session().messages

    def _cur_bookmarked() -> "set":
        return _frame()["bookmarked"]

    def _cur_branches() -> "dict":
        return _frame()["branches"]

    # ── rendering ────────────────────────────────────────────────────────

    def _row(idx: int, selected: bool) -> str:
        msgs = _cur_msgs()
        msg = msgs[idx]
        role = msg.get("role", "?")
        icon = _TREE_ROLE_ICON.get(role, "?")
        colour = _TREE_ROLE_COLOUR.get(role, "")

        raw = msg.get("content") or ""
        if isinstance(raw, list):
            raw = " ".join(p.get("text", "") for p in raw if isinstance(p, dict))
        preview = raw.replace("\n", " ").strip()[:58]

        extra_badges: "list[str]" = []
        if msg.get("tool_calls"):
            n = len(msg["tool_calls"])
            extra_badges.append(f"[{n}t]")
        if msg.get("tool_call_id"):
            extra_badges.append("[tr]")

        bm_icon = f"{_TREE_BOLD}\033[33m★\033[0m" if idx in _cur_bookmarked() else " "
        branch_count = len(_cur_branches().get(idx, []))
        br_icon = (
            f"{_TREE_CYAN}⎇{branch_count}\033[0m" if branch_count else "  "
        )
        badge_str = f" {_TREE_DIM}{''.join(extra_badges)}{_TREE_RESET}" if extra_badges else ""
        index_str = f"{idx:3}"

        if selected:
            line = (
                f"{_TREE_INVERT}{_TREE_BOLD}"
                f"  {index_str} {icon} {preview}"
                f"{_TREE_RESET} {bm_icon} {br_icon}{badge_str}"
            )
        else:
            line = (
                f"  {_TREE_DIM}{index_str}{_TREE_RESET} "
                f"{colour}{icon} {preview}{_TREE_RESET}"
                f" {bm_icon} {br_icon}{badge_str}"
            )
        return line + "\n"

    def _header() -> str:
        crumbs = []
        for i, frame in enumerate(_nav_stack):
            s = frame["session"]
            label = (s.name or s.id[:8])
            if i == len(_nav_stack) - 1:
                crumbs.append(f"{_TREE_BOLD}{_TREE_CYAN}{label}{_TREE_RESET}")
            else:
                crumbs.append(f"{_TREE_DIM}{label}{_TREE_RESET}")
        breadcrumb = f" {_TREE_DIM}→{_TREE_RESET} ".join(crumbs)

        n = len(_cur_msgs())
        bm_count = len(_cur_bookmarked())
        bm_str = f"  {_TREE_DIM}\033[33m★{bm_count}\033[0m" if bm_count else ""

        cur_s = _cur_session()
        parent_note = ""
        if cur_s.parent_id:
            parent_note = (
                f"  {_TREE_DIM}⎇ forked from "
                f"{cur_s.parent_id[:8]}@{cur_s.fork_index}{_TREE_RESET}"
            )

        keys = (
            "↑↓ move  Enter branch  f fork  b bkm  B next  "
            "→/← child/parent  q quit"
        )
        return (
            f"{_TREE_BOLD}{_TREE_CYAN}  ⎇  {_TREE_RESET}{breadcrumb}"
            f"  {_TREE_DIM}{n} msg{bm_str}{parent_note}{_TREE_RESET}\n"
            f"{_TREE_DIM}  {keys}{_TREE_RESET}\n"
            f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
        )

    def _footer() -> str:
        if mode[0] == "pick_child":
            lines = [f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"]
            lines.append(
                f"  {_TREE_BOLD}Select branch to navigate into:{_TREE_RESET}\n"
            )
            for i, child in enumerate(pick_children):
                label = child.name or child.id[:8]
                age = child.updated_at[:10] if hasattr(child, "updated_at") else ""
                sel = pick_cursor[0] == i
                entry = f"  {'▶' if sel else ' '} [{i + 1}] {child.id[:8]}  {label}  {age}"
                if sel:
                    entry = f"{_TREE_INVERT}{_TREE_BOLD}{entry}{_TREE_RESET}"
                lines.append(entry + "\n")
            lines.append(
                f"  {_TREE_DIM}↑↓ select  Enter/→ confirm  Esc/← cancel{_TREE_RESET}\n"
            )
            return "".join(lines)

        s = status[0]
        if s:
            return (
                f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
                f"  {s}\n"
            )
        idx = cursor[0]
        msgs = _cur_msgs()
        if 0 <= idx < len(msgs):
            msg = msgs[idx]
            role = msg.get("role", "?")
            raw = (msg.get("content") or "").replace("\n", " ").strip()
            bm_label = ""
            for bm in _cur_session().bookmarks:
                if bm["index"] == idx:
                    bm_label = f"  \033[33m★ {bm.get('label', '')[:40]}\033[0m"
                    break
            branches_here = _cur_branches().get(idx, [])
            br_info = ""
            if branches_here:
                names = ", ".join(b.name or b.id[:8] for b in branches_here[:3])
                extra = f" +{len(branches_here) - 3}" if len(branches_here) > 3 else ""
                br_info = f"  {_TREE_CYAN}⎇ {names}{extra}{_TREE_RESET}"
            return (
                f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n"
                f"  {_TREE_DIM}[{idx}] {role}  {raw[:60]}{_TREE_RESET}"
                f"{bm_label}{br_info}\n"
            )
        return f"{_TREE_DIM}{'─' * 72}{_TREE_RESET}\n\n"

    def _build_text() -> ANSI:
        parts = [_header()]
        for i in range(len(_cur_msgs())):
            parts.append(_row(i, i == cursor[0]))
        parts.append(_footer())
        return ANSI("".join(parts))

    # ── helpers ──────────────────────────────────────────────────────────

    def _clamp() -> None:
        total = len(_cur_msgs())
        cursor[0] = max(0, min(total - 1, cursor[0]))

    def _move(delta: int) -> None:
        if mode[0] == "pick_child":
            pick_cursor[0] = max(0, min(len(pick_children) - 1, pick_cursor[0] + delta))
        else:
            total = len(_cur_msgs())
            cursor[0] = max(0, min(total - 1, cursor[0] + delta))
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    def _next_bookmark() -> None:
        bms = sorted(_cur_bookmarked())
        if not bms:
            status[0] = f"{_TREE_DIM}no bookmarks — press 'b' to add{_TREE_RESET}"
            if app_ref[0]:
                app_ref[0].invalidate()
            return
        nxt = next((i for i in bms if i > cursor[0]), bms[0])
        cursor[0] = nxt
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    def _toggle_bookmark() -> None:
        idx = cursor[0]
        msgs = _cur_msgs()
        if idx < 0 or idx >= len(msgs):
            return
        added = sm.toggle_bookmark(_cur_session(), idx)
        _frame()["bookmarked"] = {b["index"] for b in _cur_session().bookmarks}
        action = f"\033[33m★\033[0m bookmarked [{idx}]" if added else f"removed bookmark [{idx}]"
        status[0] = action
        if app_ref[0]:
            app_ref[0].invalidate()

    def _navigate_into() -> None:
        idx = cursor[0]
        children = _cur_branches().get(idx, [])
        if not children:
            status[0] = f"{_TREE_DIM}no branches at [{idx}] — press 'f' to fork here{_TREE_RESET}"
            if app_ref[0]:
                app_ref[0].invalidate()
            return
        if len(children) == 1:
            _push_child(children[0])
        else:
            pick_children.clear()
            pick_children.extend(children)
            pick_cursor[0] = 0
            mode[0] = "pick_child"
            if app_ref[0]:
                app_ref[0].invalidate()

    def _push_child(child_meta: "object") -> None:
        try:
            child_session = sm.load(child_meta.id)
            frame = _load_frame(child_session)
            _nav_stack.append(frame)
            cursor[0] = len(child_session.messages) - 1
            status[0] = ""
        except Exception as exc:
            status[0] = f"{_TREE_RED}✗ could not load {child_meta.id[:8]}: {exc}{_TREE_RESET}"
        if app_ref[0]:
            app_ref[0].invalidate()

    def _navigate_back() -> None:
        if len(_nav_stack) <= 1:
            status[0] = f"{_TREE_DIM}already at root{_TREE_RESET}"
            if app_ref[0]:
                app_ref[0].invalidate()
            return
        _nav_stack.pop()
        _clamp()
        status[0] = ""
        if app_ref[0]:
            app_ref[0].invalidate()

    def _confirm_pick() -> None:
        if not pick_children:
            return
        child_meta = pick_children[pick_cursor[0]]
        mode[0] = "list"
        _push_child(child_meta)

    # ── key bindings ─────────────────────────────────────────────────────

    kb = KeyBindings()

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
        if mode[0] == "list":
            cursor[0] = 0
            status[0] = ""
            if app_ref[0]:
                app_ref[0].invalidate()

    @kb.add("end")
    @kb.add("G")
    def _end(_: object) -> None:
        if mode[0] == "list":
            cursor[0] = len(_cur_msgs()) - 1
            status[0] = ""
            if app_ref[0]:
                app_ref[0].invalidate()

    @kb.add("enter")
    def _enter(_: object) -> None:
        if mode[0] == "pick_child":
            _confirm_pick()
            return
        result[0] = "branch"
        if app_ref[0]:
            app_ref[0].exit()

    @kb.add("f")
    def _fork_key(_: object) -> None:
        if mode[0] == "list":
            result[0] = "fork"
            if app_ref[0]:
                app_ref[0].exit()

    @kb.add("b")
    def _bkm(_: object) -> None:
        if mode[0] == "list":
            _toggle_bookmark()

    @kb.add("B")
    def _next_bkm(_: object) -> None:
        if mode[0] == "list":
            _next_bookmark()

    @kb.add("right")
    @kb.add("l")
    def _right(_: object) -> None:
        if mode[0] == "list":
            _navigate_into()
        elif mode[0] == "pick_child":
            _confirm_pick()

    @kb.add("left")
    @kb.add("h")
    def _left(_: object) -> None:
        if mode[0] == "pick_child":
            mode[0] = "list"
            status[0] = ""
            if app_ref[0]:
                app_ref[0].invalidate()
        else:
            _navigate_back()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(_: object) -> None:
        if mode[0] == "pick_child":
            mode[0] = "list"
            status[0] = ""
            if app_ref[0]:
                app_ref[0].invalidate()
        else:
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

    # ── run in a dedicated thread (avoids conflicting event loops) ───────

    import threading as _threading

    _done = _threading.Event()
    _exc: "list[BaseException]" = []

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
    _done.wait()

    # The tree App ran full_screen in a background thread and exited the
    # alternate screen buffer.  Tell the outer REPL renderer to treat the
    # terminal as "unknown state" so the next paint re-enters the alternate
    # screen and does a full erase+redraw — the same thing that happens on
    # resize, which is why resize previously "fixed" the display.
    if reset_fn:
        try:
            reset_fn()
        except Exception:
            pass

    if _exc:
        output_fn(f"  \033[31m✗ /tree error: {_exc[0]}\033[0m\n")
        return

    # ── act on the user's choice ─────────────────────────────────────────

    chosen_idx = cursor[0]
    action = result[0]

    if action is None:
        output_fn(f"  {_TREE_DIM}/tree cancelled{_TREE_RESET}\n")
        return

    # Fork from whichever session was active at exit (top of nav stack)
    target_session = _cur_session()
    if chosen_idx >= len(target_session.messages):
        output_fn(f"  {_TREE_RED}✗ invalid index {chosen_idx}{_TREE_RESET}\n")
        return

    forked = sm.fork(target_session.id, chosen_idx)

    if action == "fork":
        output_fn(
            f"  {_TREE_GREEN}✓ forked{_TREE_RESET}"
            f"  session {_TREE_CYAN}{forked.id[:8]}{_TREE_RESET}"
            f"  {_TREE_DIM}({len(forked.messages)} msgs — live context unchanged){_TREE_RESET}\n"
        )
        return

    # action == "branch": restore agent context to the forked state
    snapshot = target_session.snapshot_at(chosen_idx)
    agent._context.restore(snapshot)
    agent._context.compactor.reset_overflow_flag()
    agent._session = forked

    output_fn(
        f"  {_TREE_GREEN}⎇  branched{_TREE_RESET}"
        f"  {_TREE_DIM}context rolled back to [{chosen_idx}]"
        f"  from {target_session.id[:8]} → {forked.id[:8]}{_TREE_RESET}\n"
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
    reset_fn: "Callable[[], None] | None" = None,
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

    if keyword == "/fork":
        if agent is None:
            _print("[dim]  /fork requires an active session.[/dim]")
            return True
        fork_name = arg or None
        sm = agent._session_manager
        msgs = [
            m.to_dict() if hasattr(m, "to_dict") else {"role": m.role, "content": m.content}
            for m in agent._context.get_messages()
        ]
        agent._session.messages = msgs
        sm.save(agent._session)
        idx = len(agent._session.messages) - 1
        if idx < 0:
            _print("[dim]  No messages to fork from.[/dim]")
            return True
        try:
            forked = sm.fork(agent._session.id, idx, name=fork_name)
            _print(Text.assemble(
                ("  ✓ forked  ", Style(color="green", bold=True)),
                (forked.id[:8], Style(color="cyan")),
                (f"  '{forked.name}'", Style(color="cyan", bold=True)),
                (f"  ({len(forked.messages)} msgs — live context unchanged)", Style(dim=True)),
            ))
        except Exception as exc:
            _print(f"[red]  ✗ fork failed: {exc}[/red]")
        return True

    if keyword == "/bookmark":
        if agent is None:
            _print("[dim]  /bookmark requires an active session.[/dim]")
            return True
        sm = agent._session_manager
        if not arg:
            # List all bookmarks
            bms = agent._session.bookmarks
            if not bms:
                _print("[dim]  No bookmarks. In /tree press 'b' to add, or: /bookmark <index> [label][/dim]")
            else:
                lines = [f"[bold]Bookmarks ({len(bms)}):[/bold]"]
                for bm in sorted(bms, key=lambda b: b["index"]):
                    lines.append(
                        f"  [cyan]{bm['index']:4}[/cyan]  [dim]{bm.get('label', '')[:70]}[/dim]"
                    )
                _print("\n".join(lines))
        else:
            bm_parts = arg.split(None, 1)
            try:
                idx = int(bm_parts[0])
                label = bm_parts[1] if len(bm_parts) > 1 else ""
                added = sm.toggle_bookmark(agent._session, idx, label)
                if added:
                    _print(Text.assemble(
                        ("  ★ bookmarked  ", Style(color="yellow", bold=True)),
                        (f"[{idx}]", Style(color="cyan")),
                    ))
                else:
                    _print(f"[dim]  removed bookmark [{idx}][/dim]")
            except ValueError:
                _print("[dim]  Usage: /bookmark <index> [label][/dim]")
        return True

    if keyword == "/copy":
        if agent is None:
            _print("[dim]  /copy requires an active session.[/dim]")
            return True
        # Find the last assistant message
        last_text = None
        for msg in reversed(agent._context.get_messages()):
            if msg.role == "assistant" and msg.content:
                last_text = msg.content
                break
        if not last_text:
            _print("[dim]  No assistant message to copy.[/dim]")
            return True
        import subprocess, platform
        try:
            _sys = platform.system()
            if _sys == "Darwin":
                proc = subprocess.run(["pbcopy"], input=last_text, text=True, check=True, timeout=5)
            elif _sys == "Linux":
                proc = subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=last_text, text=True, check=True, timeout=5,
                )
            elif _sys == "Windows":
                proc = subprocess.run(["clip"], input=last_text, text=True, check=True, timeout=5)
            else:
                _print(f"[yellow]  \u26a0 Clipboard not supported on {_sys}[/yellow]")
                return True
            preview = last_text[:80].replace("\n", " ")
            _print(Text.assemble(
                ("  \u2713 ", Style(color="green", bold=True)),
                ("copied to clipboard", Style(color="green")),
                (f"  ({len(last_text)} chars)", Style(dim=True)),
            ))
        except FileNotFoundError:
            _print("[yellow]  \u26a0 Clipboard tool not found. Install pbcopy (macOS), xclip (Linux), or clip (Windows).[/yellow]")
        except Exception as exc:
            _print(f"[red]  \u2717 copy failed: {exc}[/red]")
        return True

    if keyword == "/export":
        if agent is None:
            _print("[dim]  /export requires an active session.[/dim]")
            return True
        session = agent._session
        # Sync messages into session before export
        session.messages = [m.to_dict() if hasattr(m, "to_dict") else ({"role": m.role, "content": m.content} if hasattr(m, "role") else m) for m in agent._context.get_messages()]
        if arg and arg.endswith(".md"):
            from tau.core.session import export_session_markdown
            content = export_session_markdown(session)
        else:
            content = json.dumps(session.to_dict(), indent=2)
        if arg:
            _arg_path = Path(arg)
            if _arg_path.is_absolute() or len(_arg_path.parts) > 1:
                out_path = _arg_path.resolve()
            else:
                from tau.core.session import local_sessions_dir
                _sessions = local_sessions_dir(agent._config.workspace_root)
                _sessions.mkdir(parents=True, exist_ok=True)
                out_path = _sessions / _arg_path
            out_path.write_text(content, encoding="utf-8")
            _print(Text.assemble(
                ("  \u2713 ", Style(color="green", bold=True)),
                ("exported to ", Style(color="green")),
                (str(out_path), Style(color="cyan")),
                (f"  ({len(session.messages)} messages)", Style(dim=True)),
            ))
        else:
            _print(content)
        return True

    if keyword == "/import":
        if not arg:
            _print("[dim]  Usage: /import <path-to-session.json>[/dim]")
            return True
        _arg_src = Path(arg).expanduser()
        if _arg_src.is_absolute() or len(_arg_src.parts) > 1:
            src = _arg_src.resolve()
        else:
            from tau.core.session import local_sessions_dir
            _src_sessions = local_sessions_dir(agent._config.workspace_root if agent else ".")
            src = (_src_sessions / _arg_src).resolve()
            if not src.exists():
                # fall back to CWD
                src = _arg_src.resolve()
        if not src.exists():
            _print(f"[red]  ✗ file not found: {src}[/red]")
            return True
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            _print(f"[red]  ✗ could not parse JSON: {exc}[/red]")
            return True
        try:
            import uuid as _uuid
            from tau.core.session import Session as _Session
            session_obj = _Session.from_dict(data)
            # Assign a fresh ID to avoid colliding with an existing session
            session_obj.id = str(_uuid.uuid4())
            from tau.core.session import local_sessions_dir
            _imp_sessions = local_sessions_dir(agent._config.workspace_root if agent else ".")
            _imp_sessions.mkdir(parents=True, exist_ok=True)
            sm = SessionManager(sessions_dir=_imp_sessions)
            sm.save(session_obj)
            _print(Text.assemble(
                ("  ✓ imported: ", Style(color="green", bold=True)),
                (session_obj.id[:8], Style(color="cyan", bold=True)),
                (f"  {session_obj.name or '(unnamed)'}  ({len(session_obj.messages)} messages)",
                 Style(dim=True)),
                ("\n    resume with: ", Style(dim=True)),
                (f"tau run -s {session_obj.id[:8]}", Style(color="cyan")),
            ))
        except Exception as exc:
            _print(f"[red]  ✗ import failed: {exc}[/red]")
        return True

    if keyword == "/share":
        if agent is None:
            _print("[dim]  /share requires an active session.[/dim]")
            return True
        session = agent._session
        # Sync live context messages into session before sharing
        session.messages = [
            m.to_dict() if hasattr(m, "to_dict") else (
                {"role": m.role, "content": m.content} if hasattr(m, "role") else m
            )
            for m in agent._context.get_messages()
        ]
        fmt = "json" if arg == "--json" else "markdown"
        _print("[dim]  ↑ uploading session…[/dim]")
        try:
            from tau.core.session import share_session
            url = share_session(session, fmt=fmt)
            _print(Text.assemble(
                ("  ✓ shared: ", Style(color="green", bold=True)),
                (url, Style(color="cyan", underline=True)),
                (f"  ({len(session.messages)} messages, {fmt})", Style(dim=True)),
            ))
            # Copy URL to clipboard (best-effort; silent on failure)
            import subprocess as _sp
            import sys as _sys
            try:
                if _sys.platform == "darwin":
                    _sp.run(["pbcopy"], input=url, text=True, check=False, timeout=3)
                elif _sys.platform.startswith("linux"):
                    _sp.run(["xclip", "-selection", "clipboard"], input=url,
                            text=True, check=False, timeout=3)
            except Exception:
                pass
        except Exception as exc:
            _print(f"[red]  ✗ share failed: {exc}[/red]")
        return True

    if keyword == "/reload":

        if agent is None:
            _print("[dim]  /reload requires an active session.[/dim]")
            return True

        reloaded: list[str] = []

        # 1. Re-read config
        tau_cfg = load_config()
        theme.load(tau_cfg, force=True)
        reloaded.append("config")

        # 2. Rebuild system prompt from scratch
        workspace = agent._config.workspace_root
        base_prompt = tau_cfg.system_prompt
        from tau.context_files import load_system_prompt_override, load_context_files
        override = load_system_prompt_override(workspace)
        if override is not None:
            base_prompt = override
        # Reset system message to base (before re-injecting fragments)
        for m in agent._context._messages:
            if m.role == "system":
                m.content = base_prompt
                break
        # Re-inject context files
        ctx_text = load_context_files(workspace)
        if ctx_text:
            agent._context.inject_prompt_fragment(ctx_text)
        reloaded.append("context files")

        # 3. Clear tool registry and re-register builtins
        from tau.tools import register_builtin_tools
        agent._registry._tools.clear()
        register_builtin_tools(agent._registry)
        # Re-apply shell/fs configuration
        from tau.tools.shell import configure_shell
        from tau.tools.fs import configure_fs
        configure_shell(
            require_confirmation=tau_cfg.shell.require_confirmation,
            timeout=tau_cfg.shell.timeout,
            allowed_commands=tau_cfg.shell.allowed_commands,
            use_persistent_shell=tau_cfg.shell.use_persistent_shell,
            workspace_root=workspace,
        )
        configure_fs(workspace_root=workspace)
        # Apply tool filtering
        if tau_cfg.tools.enabled_only:
            allowed = set(tau_cfg.tools.enabled_only)
            for name in list(agent._registry.names()):
                if name not in allowed:
                    agent._registry.unregister(name)
        elif tau_cfg.tools.disabled:
            for name in tau_cfg.tools.disabled:
                agent._registry.unregister(name)

        # 4. Clear cached skill modules and reload
        import sys as _sys
        for key in [k for k in _sys.modules if k.startswith("tau_skill_")]:
            del _sys.modules[key]
        loader = SkillLoader(
            extra_paths=tau_cfg.skills.paths,
            disabled=tau_cfg.skills.disabled,
        )
        loader.load_into(agent._registry, agent._context)
        reloaded.append("skills")

        # 5. Reload extensions
        if ext_registry is not None:
            names = ext_registry.reload(
                registry=agent._registry,
                context=agent._context,
                steering=steering,
                console_print=console.print,
                disabled=tau_cfg.extensions.disabled,
            )
            reloaded.append(f"extensions ({len(names)})")

        _print(Text.assemble(
            ("  ✓ reloaded: ", Style(color="green", bold=True)),
            (", ".join(reloaded), Style(color="green")),
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
        _tree_navigator(agent, output_fn, reset_fn=reset_fn)
        return True

    if keyword == "/themes":
        from tau.config import THEME_PRESETS
        try:
            active = load_config().theme.preset
        except Exception:
            active = ""
        lines = []
        for name, colors in THEME_PRESETS.items():
            marker = " [green]◀ active[/green]" if name == active else ""
            swatch = ""
            for key in ("user_color", "assistant_color", "tool_color", "error_color", "accent_color"):
                swatch += f"[{colors[key]}]██[/{colors[key]}]"
            lines.append(f"  [{colors['accent_color']}]{name:<20}[/{colors['accent_color']}] {swatch}{marker}")
        _print("[bold]Theme presets:[/bold]  (use [cyan]/theme <name>[/cyan] to switch)\n" + "\n".join(lines))
        return True

    if keyword == "/theme":
        if not arg:
            _print("[dim]  Usage: /theme <preset-name>[/dim]")
            _print("[dim]  Use /themes to list available presets.[/dim]")
            return True
        from tau.config import THEME_PRESETS, THEME_PATH
        preset_name = arg.strip()
        if preset_name not in THEME_PRESETS:
            _print(f"[red]  Unknown preset {preset_name!r}.[/red]  Available: {', '.join(THEME_PRESETS)}")
            return True
        # Write to ~/.tau/theme.toml so it persists and hot-reload picks it up
        try:
            THEME_PATH.parent.mkdir(parents=True, exist_ok=True)
            THEME_PATH.write_text(f'preset = "{preset_name}"\n', encoding="utf-8")
            cfg = load_config()
            theme.load(cfg, force=True)
            _print(f"[green]  ✓ Switched to [bold]{preset_name}[/bold] theme.[/green]")
        except Exception as exc:  # noqa: BLE001
            _print(f"[red]  Error applying theme: {exc}[/red]")
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
    show_thinking: bool = False,
    tau_config: "TauConfig | None" = None,
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
    from prompt_toolkit.layout.dimension import Dimension as D
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
    cancel_requested = threading.Event()
    app_ref: list[Application | None] = [None]
    output_text_parts: list[str] = []
    _staged_images = staged_images or []
    _scroll_offset: list[int] = [0]  # 0 = follow bottom; >0 = lines from bottom

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
        + f"{_DIM}Shift+↑↓ scroll  ·  PgUp/PgDn page  ·  End jump to bottom  ·  Esc cancel  ·  Ctrl+J new line{_RESET}\n"
        + f"{_DIM}{'═' * 60}{_RESET}\n"
    )
    output_text_parts.append(header)

    # -- buffers -------------------------------------------------------------
    _completer = _TauCompleter()
    input_buffer = Buffer(name="input", completer=_completer, complete_while_typing=False)

    # -- helpers -------------------------------------------------------------
    # A lock + snapshot ensure that _get_output_text and _get_cursor_position
    # see a consistent view of output_text_parts during a single render cycle,
    # preventing IndexError when the spinner or /tree invalidates mid-append.
    _output_lock = threading.Lock()
    _parsed_line_count: list[int] = [1]

    def _get_output_text() -> ANSI:
        with _output_lock:
            joined = "".join(output_text_parts)
        ansi = ANSI(joined)
        # Count actual fragment lines so _get_cursor_position stays in bounds.
        from prompt_toolkit.formatted_text import to_formatted_text
        from prompt_toolkit.formatted_text.utils import split_lines
        _parsed_line_count[0] = len(list(split_lines(to_formatted_text(ansi))))
        return ansi

    def _get_cursor_position() -> Point:
        # Use the line count captured by _get_output_text (always called first
        # during a render) so y never exceeds the actual fragment_lines length.
        y = max(0, _parsed_line_count[0] - 1 - _scroll_offset[0])
        return Point(x=0, y=y)

    def _append_output(text: str) -> None:
        with _output_lock:
            output_text_parts.append(text)
        if app_ref[0]:
            app_ref[0].invalidate()

    # -- footer data ---------------------------------------------------------
    import shutil as _shutil
    _footer_tau_config = [tau_config if tau_config is not None else load_config()]

    def _get_footer_line1() -> ANSI:
        ws_name = Path(agent._config.workspace_root).name
        model = agent._config.model
        thinking = agent._config.thinking_level
        think_str = f" [{thinking}]" if thinking and thinking != "off" else ""
        left_plain = f"  {ws_name}"
        right_plain = f"{model}{think_str}  "
        try:
            width = _shutil.get_terminal_size().columns
        except Exception:
            width = 80
        pad = max(1, width - len(left_plain) - len(right_plain))
        return ANSI(
            f"  {_DIM}{ws_name}{_RESET}"
            f"{' ' * pad}"
            f"{_CYAN}{model}{_DIM}{think_str}{_RESET}  "
        )

    def _get_footer_line2() -> ANSI:
        cu = getattr(agent._session, "cumulative_usage", {})
        in_tok = cu.get("input_tokens", 0)
        out_tok = cu.get("output_tokens", 0)
        cache_r = cu.get("cache_read_tokens", 0)
        cache_w = cu.get("cache_write_tokens", 0)
        ctx_used = agent._context.token_count()
        budget = agent._config.max_tokens
        pct = int(ctx_used / budget * 100) if budget and ctx_used else 0

        class _DU:
            input_tokens = in_tok
            output_tokens = out_tok
            cache_read_tokens = cache_r
            cache_write_tokens = cache_w

        cost = _footer_tau_config[0].calculate_cost(agent._config.model, _DU())
        cost_str = f"${cost:.4f}" if cost > 0 else "$0.0000"
        pct_color = "\033[32m" if pct < 70 else "\033[33m" if pct < 90 else "\033[31m"
        if budget >= 1_000_000:
            budget_str = f"{budget // 1_000_000}M"
        elif budget >= 1_000:
            budget_str = f"{budget // 1_000}k"
        else:
            budget_str = str(budget)
        return ANSI(
            f"  {_DIM}{cost_str}"
            f"  ↑{in_tok:,}"
            f"  ↓{out_tok:,}"
            f"  {pct_color}{pct}%/{budget_str} ctx{_RESET}"
        )

    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _spinner_idx = [0]
    _spinner_timer: list[threading.Timer | None] = [None]

    def _tick_spinner() -> None:
        if agent_running.is_set() and not confirm_pending.is_set():
            _spinner_idx[0] = (_spinner_idx[0] + 1) % len(_SPINNER_FRAMES)
            if app_ref[0]:
                app_ref[0].invalidate()
            _spinner_timer[0] = threading.Timer(0.08, _tick_spinner)
            _spinner_timer[0].daemon = True
            _spinner_timer[0].start()

    def _get_prompt_prefix() -> list[tuple[str, str]]:
        if confirm_pending.is_set():
            return [("bold fg:ansiyellow", "allow? [y/N] ")]
        if agent_running.is_set():
            frame = _SPINNER_FRAMES[_spinner_idx[0]]
            return [("bold fg:ansimagenta", f"{frame} ")]
        return [("bold fg:ansicyan", "> ")]

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
        cancel_requested.clear()
        _spinner_idx[0] = 0
        _tick_spinner()
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
                show_thinking=show_thinking,
                cancel_event=cancel_requested,
            )
        except Exception as exc:
            _append_output(f"\nError: {exc}\n")
        finally:
            cancel_requested.clear()
            agent_running.clear()
            if _spinner_timer[0]:
                _spinner_timer[0].cancel()
                _spinner_timer[0] = None
            if app_ref[0]:
                app_ref[0].invalidate()

    # -- input handler -------------------------------------------------------
    def _on_enter(event: object) -> None:
        text = input_buffer.text.strip()
        input_buffer.reset()
        _scroll_offset[0] = 0  # reset scroll to bottom on new input

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
            def _tree_reset_fn() -> None:
                _a = app_ref[0]
                if _a is not None:
                    try:
                        _a.renderer.reset()
                    except Exception:
                        pass
            handled = _handle_slash(
                text, steering, ext_registry, ext_context, agent=agent, output_fn=_append_output,
                staged_images=_staged_images, reset_fn=_tree_reset_fn,
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
            _append_output(f"\n{_BOLD}{_CYAN}>{_RESET} {text}\n")
            _append_output(f"{_DIM}  📎 {n} file{'s' if n > 1 else ''} inlined: {names}{_RESET}\n")
            _append_output(f"{_DIM}{'─' * 60}{_RESET}\n")
            text = expanded
        else:
            _append_output(f"\n{_BOLD}{_CYAN}>{_RESET} {text}\n" + f"{_DIM}{'─' * 60}{_RESET}\n")
        threading.Thread(
            target=_run_agent, args=(text,), daemon=True
        ).start()

    # -- key bindings --------------------------------------------------------
    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: object) -> None:
        _on_enter(event)

    @kb.add("c-j")
    def _newline(event: object) -> None:
        """Ctrl+Enter (Ctrl+J): insert a newline into the input buffer."""
        if not agent_running.is_set() and not confirm_pending.is_set():
            input_buffer.insert_text("\n")

    @kb.add("tab")
    def _tab(event: object) -> None:
        """Trigger tab completion or cycle through completions."""
        buff = event.app.current_buffer
        if buff.complete_state:
            # Already showing completions — cycle to next
            buff.complete_next()
        else:
            buff.start_completion(select_first=False)

    @kb.add("s-up")
    def _scroll_up(event: object) -> None:
        """Scroll output up by one line."""
        max_offset = max(0, _parsed_line_count[0] - 1)
        _scroll_offset[0] = min(_scroll_offset[0] + 1, max_offset)
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("s-down")
    def _scroll_down(event: object) -> None:
        """Scroll output down by one line."""
        _scroll_offset[0] = max(0, _scroll_offset[0] - 1)
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("pageup")
    def _page_up(event: object) -> None:
        """Scroll output up by 20 lines."""
        max_offset = max(0, _parsed_line_count[0] - 1)
        _scroll_offset[0] = min(_scroll_offset[0] + 20, max_offset)
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("pagedown")
    def _page_down(event: object) -> None:
        """Scroll output down by 20 lines."""
        _scroll_offset[0] = max(0, _scroll_offset[0] - 20)
        if app_ref[0]:
            app_ref[0].invalidate()

    @kb.add("end")
    def _scroll_bottom(event: object) -> None:
        """Scroll to the bottom (latest output)."""
        _scroll_offset[0] = 0
        if app_ref[0]:
            app_ref[0].invalidate()

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

    @kb.add("escape")
    def _escape(event: object) -> None:
        """Escape: cancel the running agent stream."""
        if agent_running.is_set():
            cancel_requested.set()
            if app_ref[0]:
                app_ref[0].invalidate()

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
        height=D(min=1, max=10),
        wrap_lines=True,
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
        Window(height=1, char="─", style="class:separator"),
        Window(height=1, content=FormattedTextControl(_get_footer_line1, focusable=False)),
        Window(height=1, content=FormattedTextControl(_get_footer_line2, focusable=False)),
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
        mouse_support=False,  # disabled so the terminal can handle native text selection
    )
    app_ref[0] = application
    _theme_watcher = _ThemeWatcher(app_ref)
    _theme_watcher.start()
    try:
        application.run()
    finally:
        _theme_watcher.stop()


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
@click.argument("args", nargs=-1, required=False)
@_agent_options
def run_cmd(
    args: tuple[str, ...],
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
    show_thinking: bool,
    output_mode: str | None,
    print_mode: bool,
    template_name: str | None,
    var: tuple[str, ...],
    max_cost: float | None,
    no_session: bool,
    trace_log: str | None,
) -> None:
    """Run the agent (REPL if no PROMPT given, single-shot otherwise)."""
    # Split args into @file tokens and plain text tokens.
    # e.g. `tau run @code.py @tests.py "review these"` →
    #      file_args=["@code.py", "@tests.py"], prompt="review these"
    file_args = [a for a in args if a.startswith("@")]
    text_args  = [a for a in args if not a.startswith("@")]
    prompt: str | None = " ".join(text_args) if text_args else None
    if file_args:
        file_str = " ".join(file_args)
        prompt = (file_str + "\n\n" + prompt) if prompt else file_str
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
    # Resolve trace log path: flag without value → default in workspace
    if trace_log == "__default__":
        trace_log = str(Path(workspace).resolve() / "tau-trace.log")
    from tau.core.trace import configure_trace
    configure_trace(trace_log)
    if trace_log:
        console.print(f"[dim]Trace logging to: {trace_log}[/dim]")
    ensure_tau_home()
    tau_config = load_config()
    theme.load(tau_config)
    agent_config = _make_agent_config(tau_config, provider, model, think, no_confirm, workspace, no_parallel, persistent_shell, max_cost)
    if no_session:
        from tau.sdk import InMemorySessionManager
        session_manager = InMemorySessionManager()
    else:
        from tau.core.session import local_sessions_dir
        session_manager = SessionManager(sessions_dir=local_sessions_dir(workspace))
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
        _render_events(agent, prompt, verbose, images=list(image) if image else None, ext_registry=ext_registry, show_thinking=show_thinking)
    else:
        _repl(agent, agent_config, verbose, ext_registry=ext_registry, staged_images=list(image) if image else None, show_thinking=show_thinking, tau_config=tau_config)

# ---------------------------------------------------------------------------
# `tau sessions` subcommands
# ---------------------------------------------------------------------------
@main.group("sessions")
def sessions_group() -> None:
    """Manage saved sessions."""

@sessions_group.command("list")
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_list(workspace: str) -> None:
    """List sessions in the current project workspace."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
    metas = sm.list_sessions()
    sdir = local_sessions_dir(workspace)
    if not metas:
        console.print(f"[dim]No sessions found in {sdir}[/dim]")
        return
    console.print(f"[dim]Sessions in {sdir}:[/dim]")
    console.print(f"[bold]{'ID':10} {'NAME':24} {'MODEL':22} {'UPDATED':19}[/bold]")
    console.print(Rule(style="dim"))
    for m in metas:
        console.print(m.display())

@sessions_group.command("show")
@click.argument("session_id")
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_show(session_id: str, workspace: str) -> None:
    """Show details and messages for a session."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
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
        title="Session", border_style=theme.accent_color,
    ))
    for i, msg in enumerate(session.messages):
        role = msg.get("role", "?")
        content = (msg.get("content") or "")[:200]
        style = {"user": theme.user_color, "assistant": theme.assistant_color, "tool": theme.tool_color, "system": theme.system_color}.get(role, "white")
        console.print(f"[{style}][{i}] {role}:[/{style}] {content}")

@sessions_group.command("delete")
@click.argument("session_id")
@click.option("--yes", "-y", is_flag=True, default=False)
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_delete(session_id: str, yes: bool, workspace: str) -> None:
    """Delete a session from the project workspace."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
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
    max_cost: float | None, no_session: bool,
) -> None:
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
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
        agent_config = _make_agent_config(tau_config, provider, model, think, no_confirm, workspace, no_parallel, max_cost=max_cost)
        steering = SteeringChannel()
        agent, ext_reg = _build_agent(
            tau_config=tau_config, agent_config=agent_config,
            session_manager=sm, session_name=None, resume_id=forked.id, steering=steering,
        )
        _repl(agent, agent_config, verbose, ext_registry=ext_reg)

@sessions_group.command("branches")
@click.argument("session_id")
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_branches(session_id: str, workspace: str) -> None:
    """List all sessions forked from a given session."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
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
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_fork_points(session_id: str, workspace: str) -> None:
    """List fork points (user messages) in a session."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
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

@sessions_group.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--resume", "-r", is_flag=True, default=False, help="Start REPL in the imported session immediately.")
@_agent_options
def sessions_import(
    file: str,
    resume: bool,
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
    show_thinking: bool,
    output_mode: str | None,
    print_mode: bool,
    template_name: str | None,
    var: tuple[str, ...],
    max_cost: float | None,
    no_session: bool,
    trace_log: str | None,
) -> None:
    """Import a previously-exported JSON session file."""
    import uuid
    from tau.core.session import Session
    src = Path(file).resolve()
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        err_console.print(f"[red]Error reading file:[/red] {exc}")
        sys.exit(1)
    try:
        session_obj = Session.from_dict(data)
    except Exception as exc:
        err_console.print(f"[red]Error parsing session JSON:[/red] {exc}")
        sys.exit(1)
    # Fresh ID to avoid collision with any existing session
    session_obj.id = str(uuid.uuid4())
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
    sm.save(session_obj)
    sdir = local_sessions_dir(workspace)
    console.print(Text.assemble(
        ("  ✓ imported: ", Style(color="green", bold=True)),
        (session_obj.id[:8], Style(color="cyan", bold=True)),
        (f"  {session_obj.name or '(unnamed)'}  ({len(session_obj.messages)} messages)",
         Style(dim=True)),
    ))
    console.print(f"[dim]  Saved to: {sdir}[/dim]")
    console.print(f"[dim]  Resume: [cyan]tau run -s {session_obj.id[:8]} -w {workspace}[/cyan][/dim]")
    if resume:
        _setup_logging(verbose)
        ensure_tau_home()
        tau_config = load_config()
        agent_config = _make_agent_config(
            tau_config, provider, model, think, no_confirm, workspace, no_parallel,
            persistent_shell, max_cost=max_cost,
        )
        steering = SteeringChannel()
        agent, ext_reg = _build_agent(
            tau_config=tau_config, agent_config=agent_config,
            session_manager=sm, session_name=None, resume_id=session_obj.id,
            steering=steering,
        )
        _repl(agent, agent_config, verbose, ext_registry=ext_reg)

@sessions_group.command("export")
@click.argument("session_id")
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "markdown"]), default="json", help="Export format.")
@click.option("--output", "-o", "output_file", default=None, help="Write to file instead of stdout.")
@click.option("--workspace", "-w", default=".", show_default=True, help="Project workspace root.")
def sessions_export(session_id: str, fmt: str, output_file: str | None, workspace: str) -> None:
    """Export a session as JSON or Markdown."""
    from tau.core.session import local_sessions_dir
    sm = SessionManager(sessions_dir=local_sessions_dir(workspace))
    try:
        session = sm.load(session_id)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    if fmt == "markdown":
        from tau.core.session import export_session_markdown
        content = export_session_markdown(session)
    else:
        content = json.dumps(session.to_dict(), indent=2)
    if output_file:
        Path(output_file).write_text(content, encoding="utf-8")
        console.print(Text.assemble(
            ("  \u2713 ", Style(color="green", bold=True)),
            ("exported to ", Style(color="green")),
            (output_file, Style(color="cyan")),
        ))
    else:
        click.echo(content)

# ---------------------------------------------------------------------------
# `tau themes` subcommands
# ---------------------------------------------------------------------------
@main.group("themes")
def themes_group() -> None:
    """Manage and preview built-in theme presets."""


@themes_group.command("list")
def themes_list() -> None:
    """List all built-in theme presets with a colour preview."""
    from tau.config import THEME_PRESETS, THEME_PATH, ThemeConfig, load_config
    try:
        active_cfg = load_config()
        active_preset = active_cfg.theme.preset
    except Exception:
        active_preset = ""

    console.print()
    console.print(Rule("Built-in theme presets", style="dim"))
    console.print(
        "[dim]  Use in [bold]~/.tau/theme.toml[/bold]:[/dim]"
        "  [cyan]preset = \"<name>\"[/cyan]"
    )
    console.print()

    for name, colors in THEME_PRESETS.items():
        marker = " ◀ active" if name == active_preset else ""
        console.print(Text.assemble(
            ("  ", ""),
            (f"{name:<20}", Style(color=colors["accent_color"], bold=True)),
            (marker, Style(color="green", bold=True, dim=not marker)),
        ))
        # Colour swatches
        swatch_parts: list[tuple[str, str]] = [("    ", "")]
        for label, key in (
            ("user", "user_color"),
            ("assistant", "assistant_color"),
            ("tool", "tool_color"),
            ("error", "error_color"),
            ("accent", "accent_color"),
        ):
            swatch_parts.append((f"  {label} ", "dim"))
            swatch_parts.append(("██", colors[key]))
        console.print(Text.assemble(*swatch_parts))
    console.print()


@themes_group.command("show")
@click.argument("preset_name")
def themes_show(preset_name: str) -> None:
    """Show all colour values for a specific preset."""
    from tau.config import THEME_PRESETS
    if preset_name not in THEME_PRESETS:
        err_console.print(
            f"[red]Unknown preset {preset_name!r}.[/red]  "
            f"Available: {', '.join(THEME_PRESETS)}"
        )
        sys.exit(1)
    colors = THEME_PRESETS[preset_name]
    console.print()
    console.print(Rule(f"Theme: {preset_name}", style="dim"))
    for field, value in colors.items():
        console.print(Text.assemble(
            (f"  {field:<20}", Style(dim=True)),
            ("██  ", value),
            (value, Style(color=value)),
        ))
    console.print()


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
