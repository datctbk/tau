"""Extension base class and registry for tau."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tau.core.types import ExtensionManifest, SlashCommand, ToolDefinition

if TYPE_CHECKING:
    from tau.core.context import ContextManager
    from tau.core.steering import SteeringChannel
    from tau.core.tool_registry import ToolRegistry
    from tau.core.types import (
        AfterToolCallContext,
        AfterToolCallResult,
        BeforeToolCallContext,
        BeforeToolCallResult,
        Event,
    )

logger = logging.getLogger(__name__)

# Type alias for event hooks: receives every Event yielded by agent.run()
EventHook = Callable[["Event"], None]


# ---------------------------------------------------------------------------
# Extension base class
# ---------------------------------------------------------------------------

class Extension:
    """
    Base class for all tau extensions.

    Subclass this, set ``manifest``, and override whichever hooks you need.
    The extension file must expose a module-level ``EXTENSION`` instance.

    Minimal example::

        class MyExtension(Extension):
            manifest = ExtensionManifest(name="my_ext", description="Does stuff.")

            def tools(self) -> list[ToolDefinition]:
                return [ToolDefinition(...)]

        EXTENSION = MyExtension()
    """

    #: Subclasses MUST set this as a class attribute.
    manifest: ExtensionManifest

    # ------------------------------------------------------------------
    # Override these to add capabilities
    # ------------------------------------------------------------------

    def tools(self) -> list[ToolDefinition]:
        """Return ToolDefinitions to register in the ToolRegistry."""
        return []

    def slash_commands(self) -> list[SlashCommand]:
        """Return slash commands this extension handles."""
        return []

    def handle_slash(self, command: str, args: str, context: "ExtensionContext") -> bool:
        """
        Handle a /command from the REPL.

        ``command`` is the keyword without the slash (e.g. ``"fmt"``).
        ``args`` is everything after the keyword (stripped).
        Return True if handled, False to fall through.
        """
        return False

    def event_hook(self, event: "Event") -> None:
        """Called for every Event yielded by agent.run(). Override to observe/react."""
        pass

    def before_tool_call(self, context: "BeforeToolCallContext") -> "BeforeToolCallResult | None":
        """Called before a tool executes. Return a result to block execution."""
        return None

    def after_tool_call(self, context: "AfterToolCallContext") -> "AfterToolCallResult | None":
        """Called after a tool executes but before yielding the result. Return a result to override."""
        return None

    def on_load(self, context: "ExtensionContext") -> None:
        """Called once after the extension is fully registered."""
        pass

    def on_unload(self) -> None:
        """Called when the extension is removed at runtime (future use)."""
        pass


# ---------------------------------------------------------------------------
# Context passed to extensions at runtime
# ---------------------------------------------------------------------------

class ExtensionContext:
    """
    Thin façade exposing the subset of tau internals that extensions may use.
    Avoids giving extensions direct access to the full Agent.
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        context: "ContextManager",
        steering: "SteeringChannel | None",
        console_print: Callable[[str], None],
        agent_config: "AgentConfig | None" = None,
    ) -> None:
        self._registry = registry
        self._context = context
        self._steering = steering
        self._console_print = console_print
        self._agent_config = agent_config
        self._pause_spinner: Callable[[], None] | None = None
        self._resume_spinner: Callable[[], None] | None = None
        self._set_spinner: Callable[..., None] | None = None

    # --- tool registry ---

    def register_tool(self, tool: ToolDefinition) -> None:
        self._registry.register(tool)

    def registered_tools(self) -> list[str]:
        return self._registry.names()

    # --- messaging ---

    def enqueue(self, message: str) -> None:
        """Add a follow-up prompt to the steering queue."""
        if self._steering is not None:
            self._steering.enqueue(message)

    def print(self, content: Any, **kwargs: Any) -> None:
        """Print to the REPL console (rich markup / renderables supported).

        *content* may be a plain string (with Rich markup) or any Rich
        renderable such as ``Panel``, ``Markdown``, ``Table``, etc.

        Extra *kwargs* (e.g. ``end=""``) are forwarded to the
        underlying console callback when it accepts them.
        """
        if kwargs:
            try:
                self._console_print(content, **kwargs)
            except TypeError:
                self._console_print(content)
        else:
            self._console_print(content)

    def pause_spinner(self) -> None:
        """Pause the CLI spinner (no-op if not wired)."""
        if self._pause_spinner is not None:
            self._pause_spinner()

    def resume_spinner(self) -> None:
        """Resume the CLI spinner (no-op if not wired)."""
        if self._resume_spinner is not None:
            self._resume_spinner()

    def set_spinner(self, msg: str, key: str = "_default") -> None:
        """Update the spinner message (no-op if not wired).

        *key* identifies the source so multiple concurrent agents
        can each show their own status line without clobbering.
        Pass an empty *msg* to remove the status for *key*.
        """
        if self._set_spinner is not None:
            self._set_spinner(msg, key=key)

    # --- context inspection ---

    def token_count(self) -> int:
        return self._context.token_count()

    # --- sub-agent spawning ---

    def create_sub_session(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        workspace: str | None = None,
        session_name: str | None = None,
        max_turns: int | None = None,
        load_skills: bool = False,
        load_extensions: bool = False,
        load_context_files: bool = False,
        shell_confirm: bool = False,
        allowed_tools: list[str] | None = None,
        max_tool_result_chars: int = 0,
    ) -> Any:
        """Spawn a child agent session for sub-agent tools.

        By default inherits the parent's provider, model, and workspace.
        Override any parameter to customise the child agent.

        Returns a ``TauSession`` (from ``tau.sdk``) ready for
        ``prompt()`` / ``prompt_sync()`` calls.  Use as a context
        manager for automatic cleanup::

            with ext_context.create_sub_session(
                system_prompt="You are a research assistant."
            ) as sub:
                events = sub.prompt_sync("Find all TODO comments")
        """
        from tau.sdk import create_session
        from tau.core.types import AgentConfig

        # Build child config inheriting from parent where not overridden
        parent = self._agent_config
        child_kwargs: dict[str, Any] = {
            "in_memory": True,
            "load_skills": load_skills,
            "load_extensions": load_extensions,
            "load_context_files": load_context_files,
            "shell_confirm": shell_confirm,
        }

        child_kwargs["provider"] = provider or (parent.provider if parent else None)
        child_kwargs["model"] = model or (parent.model if parent else None)
        child_kwargs["workspace"] = workspace or (parent.workspace_root if parent else ".")

        if system_prompt is not None:
            child_kwargs["system_prompt"] = system_prompt
        if session_name is not None:
            child_kwargs["session_name"] = session_name

        if max_turns is not None:
            config = AgentConfig()
            config.max_turns = max_turns
            child_kwargs["config"] = config

        if allowed_tools is not None:
            child_kwargs["allowed_tools"] = allowed_tools

        if max_tool_result_chars > 0:
            child_kwargs["max_tool_result_chars"] = max_tool_result_chars

        return create_session(**child_kwargs)


# ---------------------------------------------------------------------------
# ExtensionRegistry
# ---------------------------------------------------------------------------

class ExtensionRegistry:
    """
    Discovers, loads, and manages Extension instances.

    Search order (first match wins for same name):
      1. Built-in  tau/extensions/
      2. User      ~/.tau/extensions/
      3. Extra     paths from config [extensions] paths = [...]
    """

    def __init__(
        self,
        extra_paths: list[str] | None = None,
        disabled: list[str] | None = None,
        include_builtins: bool = True,
    ) -> None:
        from tau.core.session import TAU_HOME  # avoid circular import at module level
        _builtin_dir = Path(__file__).parent.parent / "extensions"

        self._search_paths: list[Path] = []
        if include_builtins:
            self._search_paths.append(_builtin_dir)
            self._search_paths.append(TAU_HOME / "extensions")

        # Auto-discover extensions from installed packages
        try:
            from tau.packages import PackageManager
            pm = PackageManager()
            for pkg_path in pm.get_extension_paths():
                self._search_paths.append(Path(pkg_path))
        except Exception:  # noqa: BLE001
            logger.debug("Could not load package manager paths (not fatal).")

        for p in (extra_paths or []):
            self._search_paths.append(Path(p).expanduser())

        self._disabled: set[str] = set(disabled or [])
        self._extensions: dict[str, Extension] = {}   # name → instance
        self._slash_index: dict[str, Extension] = {}  # command → extension
        self._hooks: list[EventHook] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(
        self,
        registry: "ToolRegistry",
        context: "ContextManager",
        steering: "SteeringChannel | None",
        console_print: Callable[[str], None],
        agent_config: "AgentConfig | None" = None,
    ) -> list[str]:
        """
        Discover and load all extensions.  Returns list of loaded names.
        Errors are logged as warnings; bad extensions are skipped.
        """
        ext_context = ExtensionContext(
            registry, context, steering, console_print,
            agent_config=agent_config,
        )
        self._ext_context = ext_context  # keep reference for CLI patching
        loaded: list[str] = []

        seen: set[str] = set()
        for base in self._search_paths:
            if not base.is_dir():
                continue
            for candidate in sorted(base.iterdir()):
                ext = self._load_one(candidate)
                if ext is None:
                    continue
                name = ext.manifest.name
                if name in self._disabled:
                    logger.debug("Extension %r disabled — skipping.", name)
                    continue
                if name in seen:
                    logger.debug("Extension %r already loaded — skipping duplicate.", name)
                    continue
                seen.add(name)
                self._register(ext, registry, context, ext_context)
                loaded.append(name)

        return loaded

    def _load_one(self, candidate: Path) -> Extension | None:
        """
        Try to load an Extension from *candidate* (a file or directory).

        Supported layouts:
          - ``my_ext.py``          — single-file extension
          - ``my_ext/``            — package; must contain ``extension.py``
          - ``my_ext/__init__.py`` — package; __init__.py exposes EXTENSION
        """
        if candidate.is_file() and candidate.suffix == ".py" and not candidate.name.startswith("_"):
            return self._import_extension(candidate)

        if candidate.is_dir():
            # Prefer explicit extension.py, fall back to __init__.py
            for fname in ("extension.py", "__init__.py"):
                entry = candidate / fname
                if entry.exists():
                    return self._import_extension(entry)

        return None

    def _import_extension(self, path: Path) -> Extension | None:
        module_name = f"_tau_ext_{path.stem}_{abs(hash(path))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to import extension from %s: %s", path, exc)
            return None

        ext = getattr(mod, "EXTENSION", None)
        if ext is None:
            logger.debug("%s has no EXTENSION — skipping.", path)
            return None
        if not isinstance(ext, Extension):
            logger.warning("%s: EXTENSION is not an Extension instance — skipping.", path)
            return None
        if not hasattr(ext, "manifest") or not isinstance(ext.manifest, ExtensionManifest):
            logger.warning("%s: EXTENSION has no valid manifest — skipping.", path)
            return None

        logger.debug("Loaded extension %r from %s", ext.manifest.name, path)
        return ext

    def _register(
        self,
        ext: Extension,
        registry: "ToolRegistry",
        context: "ContextManager",
        ext_context: ExtensionContext,
    ) -> None:
        name = ext.manifest.name
        self._extensions[name] = ext

        # Tools — call once, cache result to avoid double-invocation
        try:
            tools = ext.tools()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extension %r tools() raised: %s", name, exc)
            tools = []
        for tool in tools:
            registry.register(tool)

        # System prompt fragment
        if ext.manifest.system_prompt_fragment:
            context.inject_prompt_fragment(ext.manifest.system_prompt_fragment)

        # Slash commands
        for cmd in ext.slash_commands():
            if cmd.name in self._slash_index:
                logger.warning(
                    "Extension %r: slash command /%s already registered by %r — overwriting.",
                    name, cmd.name, self._slash_index[cmd.name].manifest.name,
                )
            self._slash_index[cmd.name] = ext

        # Event hook — only register if the subclass actually overrides event_hook
        if type(ext).event_hook is not Extension.event_hook:
            self._hooks.append(ext.event_hook)

        # on_load callback
        try:
            ext.on_load(ext_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extension %r on_load() raised: %s", name, exc)

        logger.debug(
            "Registered extension %r — %d tools, %d slash commands, hook=%s",
            name, len(tools), len(ext.slash_commands()),
            type(ext).event_hook is not Extension.event_hook,
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def reload(
        self,
        registry: "ToolRegistry",
        context: "ContextManager",
        steering: "SteeringChannel | None",
        console_print: Callable[[str], None],
        disabled: list[str] | None = None,
        agent_config: "AgentConfig | None" = None,
    ) -> list[str]:
        """Unload all extensions, clear cached modules, and re-discover from disk."""
        for ext in self._extensions.values():
            try:
                ext.on_unload()
            except Exception:  # noqa: BLE001
                pass
        # Remove cached extension modules so importlib re-reads the files
        for key in [k for k in sys.modules if k.startswith("_tau_ext_")]:
            del sys.modules[key]
        self._extensions.clear()
        self._slash_index.clear()
        self._hooks.clear()
        if disabled is not None:
            self._disabled = set(disabled)
        return self.load_all(registry, context, steering, console_print, agent_config=agent_config)

    def handle_slash(self, raw_input: str, ext_context: ExtensionContext) -> bool:
        """
        Try to dispatch a /command to a registered extension.
        Returns True if an extension handled it.
        """
        parts = raw_input.lstrip("/").split(None, 1)
        command = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""
        ext = self._slash_index.get(command)
        if ext is None:
            return False
        try:
            return ext.handle_slash(command, args, ext_context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extension %r handle_slash raised: %s", ext.manifest.name, exc)
            return False

    def fire_hooks(self, event: "Event") -> None:
        """Call every registered event hook with *event*."""
        for hook in self._hooks:
            try:
                hook(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extension event hook raised: %s", exc)

    def fire_before_tool_call(self, context: "BeforeToolCallContext") -> "BeforeToolCallResult | None":
        """Dispatch before_tool_call hook to all extensions. Short-circuits if blocked."""
        for ext in self._extensions.values():
            try:
                if type(ext).before_tool_call is not Extension.before_tool_call:
                    res = ext.before_tool_call(context)
                    if res is not None and res.block:
                        return res
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extension before_tool_call raised: %s", exc)
        return None

    def fire_after_tool_call(self, context: "AfterToolCallContext") -> None:
        """Dispatch after_tool_call hook, applying any content/is_error mutations to context.result in place."""
        for ext in self._extensions.values():
            try:
                if type(ext).after_tool_call is not Extension.after_tool_call:
                    res = ext.after_tool_call(context)
                    if res is not None:
                        if res.content is not None:
                            context.result.content = res.content
                        if res.is_error is not None:
                            context.result.is_error = res.is_error
            except Exception as exc:  # noqa: BLE001
                logger.warning("Extension after_tool_call raised: %s", exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def loaded_extensions(self) -> list[ExtensionManifest]:
        return [ext.manifest for ext in self._extensions.values()]

    def all_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of (command, description) for every registered slash command."""
        result: list[tuple[str, str]] = []
        for cmd_name, ext in self._slash_index.items():
            cmds = {c.name: c for c in ext.slash_commands()}
            desc = cmds[cmd_name].description if cmd_name in cmds else ""
            result.append((cmd_name, desc))
        return sorted(result)

    def get(self, name: str) -> Extension | None:
        return self._extensions.get(name)

    def __len__(self) -> int:
        return len(self._extensions)
