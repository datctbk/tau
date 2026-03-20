"""Tests for the Extension system (Feature #4)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tau.core.context import ContextManager
from tau.core.extension import Extension, ExtensionContext, ExtensionRegistry
from tau.core.session import SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ExtensionLoadError,
    ExtensionManifest,
    ProviderResponse,
    SlashCommand,
    TextDelta,
    TokenUsage,
    ToolDefinition,
    ToolParameter,
    TurnComplete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> AgentConfig:
    return AgentConfig(
        provider="openai",
        model="gpt-4o",
        compaction_enabled=False,
        retry_enabled=False,
    )


def _make_registry(tmp_path: Path, extra_paths: list[str] | None = None, disabled: list[str] | None = None) -> ExtensionRegistry:
    return ExtensionRegistry(
        extra_paths=extra_paths or [str(tmp_path)],
        disabled=disabled or [],
        include_builtins=False,
    )


def _load(reg: ExtensionRegistry, tmp_path: Path | None = None) -> list[str]:
    r = ToolRegistry()
    c = ContextManager(_cfg())
    return reg.load_all(r, c, steering=None, console_print=lambda _: None)


def _write_ext(tmp_path: Path, filename: str, code: str) -> Path:
    p = tmp_path / filename
    p.write_text(code, encoding="utf-8")
    return p


def _minimal_ext(name: str = "my_ext", extra: str = "") -> str:
    return f"""\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest

class _Ext(Extension):
    manifest = ExtensionManifest(name={name!r}, description="test")
    {extra}

EXTENSION = _Ext()
"""


def _ok(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text, tool_calls=[], stop_reason="end_turn",
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


# ===========================================================================
# ExtensionManifest / SlashCommand types
# ===========================================================================

class TestExtensionManifest:
    def test_required_name(self):
        m = ExtensionManifest(name="foo")
        assert m.name == "foo"

    def test_defaults(self):
        m = ExtensionManifest(name="x")
        assert m.version == "0.1.0"
        assert m.description == ""
        assert m.author == ""
        assert m.system_prompt_fragment == ""

    def test_full_fields(self):
        m = ExtensionManifest(
            name="x", version="1.2.3", description="desc",
            author="alice", system_prompt_fragment="extra prompt",
        )
        assert m.version == "1.2.3"
        assert m.author == "alice"
        assert m.system_prompt_fragment == "extra prompt"


class TestSlashCommand:
    def test_required_name(self):
        cmd = SlashCommand(name="fmt")
        assert cmd.name == "fmt"

    def test_defaults(self):
        cmd = SlashCommand(name="fmt")
        assert cmd.description == ""
        assert cmd.usage == ""

    def test_full_fields(self):
        cmd = SlashCommand(name="fmt", description="format code", usage="/fmt [file]")
        assert cmd.description == "format code"
        assert cmd.usage == "/fmt [file]"


# ===========================================================================
# Extension base class
# ===========================================================================

class TestExtensionBaseClass:
    def test_tools_returns_empty_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        assert E().tools() == []

    def test_slash_commands_returns_empty_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        assert E().slash_commands() == []

    def test_handle_slash_returns_false_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        ctx = MagicMock(spec=ExtensionContext)
        assert E().handle_slash("cmd", "args", ctx) is False

    def test_event_hook_noop_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        # Should not raise
        E().event_hook(TurnComplete(usage=TokenUsage()))

    def test_on_load_noop_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        ctx = MagicMock(spec=ExtensionContext)
        E().on_load(ctx)   # no raise

    def test_on_unload_noop_by_default(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
        E().on_unload()    # no raise

    def test_subclass_can_provide_tools(self):
        tool = ToolDefinition(
            name="t", description="d",
            parameters={"x": ToolParameter(type="string", description="x")},
            handler=lambda x: x,
        )
        class E(Extension):
            manifest = ExtensionManifest(name="e")
            def tools(self): return [tool]
        assert E().tools() == [tool]

    def test_subclass_can_provide_slash_commands(self):
        class E(Extension):
            manifest = ExtensionManifest(name="e")
            def slash_commands(self): return [SlashCommand(name="foo")]
        assert E().slash_commands()[0].name == "foo"


# ===========================================================================
# ExtensionContext
# ===========================================================================

class TestExtensionContext:
    def _ctx(self) -> tuple[ExtensionContext, ToolRegistry, ContextManager]:
        reg = ToolRegistry()
        ctx = ContextManager(_cfg())
        steering = SteeringChannel()
        prints: list[str] = []
        ec = ExtensionContext(
            registry=reg,
            context=ctx,
            steering=steering,
            console_print=prints.append,
        )
        ec._prints = prints
        return ec, reg, ctx

    def test_register_tool(self):
        ec, reg, _ = self._ctx()
        tool = ToolDefinition(
            name="t", description="d",
            parameters={},
            handler=lambda: "ok",
        )
        ec.register_tool(tool)
        assert "t" in reg.names()

    def test_registered_tools_returns_names(self):
        ec, reg, _ = self._ctx()
        tool = ToolDefinition(name="t2", description="d", parameters={}, handler=lambda: "ok")
        reg.register(tool)
        assert "t2" in ec.registered_tools()

    def test_enqueue_adds_to_steering(self):
        ec, _, _ = self._ctx()
        ec.enqueue("follow up")
        assert ec._steering.queue_size() == 1

    def test_enqueue_no_op_without_steering(self):
        reg = ToolRegistry()
        ctx = ContextManager(_cfg())
        ec = ExtensionContext(registry=reg, context=ctx, steering=None, console_print=lambda _: None)
        ec.enqueue("x")   # should not raise

    def test_print_calls_console(self):
        ec, _, _ = self._ctx()
        ec.print("hello rich")
        assert "hello rich" in ec._prints

    def test_token_count_nonzero_after_messages(self):
        ec, _, ctx = self._ctx()
        from tau.core.types import Message
        ctx.add_message(Message(role="user", content="hello world"))
        assert ec.token_count() > 0


# ===========================================================================
# ExtensionRegistry — discovery & loading
# ===========================================================================

class TestExtensionRegistryDiscovery:
    def test_empty_dir_loads_nothing(self, tmp_path):
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert loaded == []

    def test_loads_single_file_extension(self, tmp_path):
        _write_ext(tmp_path, "my_ext.py", _minimal_ext("my_ext"))
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert "my_ext" in loaded

    def test_skips_files_without_extension_attr(self, tmp_path):
        (tmp_path / "no_ext.py").write_text("x = 1\n", encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert loaded == []

    def test_skips_files_with_wrong_extension_type(self, tmp_path):
        (tmp_path / "bad.py").write_text("EXTENSION = 42\n", encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert loaded == []

    def test_skips_private_files(self, tmp_path):
        _write_ext(tmp_path, "_private.py", _minimal_ext("_private"))
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert loaded == []

    def test_loads_package_extension_via_extension_py(self, tmp_path):
        pkg = tmp_path / "my_pkg"
        pkg.mkdir()
        (pkg / "extension.py").write_text(_minimal_ext("pkg_ext"), encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert "pkg_ext" in loaded

    def test_loads_package_extension_via_init_py(self, tmp_path):
        pkg = tmp_path / "my_pkg2"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(_minimal_ext("init_ext"), encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert "init_ext" in loaded

    def test_extension_py_preferred_over_init_py(self, tmp_path):
        pkg = tmp_path / "prefer_test"
        pkg.mkdir()
        (pkg / "extension.py").write_text(_minimal_ext("from_ext_py"), encoding="utf-8")
        (pkg / "__init__.py").write_text(_minimal_ext("from_init_py"), encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert "from_ext_py" in loaded
        assert "from_init_py" not in loaded

    def test_disabled_extension_not_loaded(self, tmp_path):
        _write_ext(tmp_path, "skip_me.py", _minimal_ext("skip_me"))
        reg = _make_registry(tmp_path, disabled=["skip_me"])
        loaded = _load(reg)
        assert "skip_me" not in loaded

    def test_duplicate_name_skipped(self, tmp_path):
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        dir1.mkdir(); dir2.mkdir()
        _write_ext(dir1, "dup.py", _minimal_ext("dup_ext"))
        _write_ext(dir2, "dup.py", _minimal_ext("dup_ext"))
        reg = ExtensionRegistry(
            extra_paths=[str(dir1), str(dir2)],
            include_builtins=False,
        )
        loaded = _load(reg)
        assert loaded.count("dup_ext") == 1

    def test_syntax_error_in_file_skipped(self, tmp_path):
        (tmp_path / "broken.py").write_text("def (: broken syntax\n", encoding="utf-8")
        reg = _make_registry(tmp_path)
        loaded = _load(reg)   # should not raise
        assert loaded == []

    def test_returns_list_of_loaded_names(self, tmp_path):
        _write_ext(tmp_path, "alpha.py", _minimal_ext("alpha"))
        _write_ext(tmp_path, "beta.py", _minimal_ext("beta"))
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        assert set(loaded) == {"alpha", "beta"}

    def test_len_reflects_loaded_count(self, tmp_path):
        _write_ext(tmp_path, "one.py", _minimal_ext("one"))
        _write_ext(tmp_path, "two.py", _minimal_ext("two"))
        reg = _make_registry(tmp_path)
        _load(reg)
        assert len(reg) == 2


# ===========================================================================
# ExtensionRegistry — tool registration
# ===========================================================================

class TestExtensionRegistryTools:
    def test_tools_registered_in_tool_registry(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest, ToolDefinition, ToolParameter

class _E(Extension):
    manifest = ExtensionManifest(name="tool_ext")
    def tools(self):
        return [ToolDefinition(
            name="my_tool", description="d",
            parameters={"x": ToolParameter(type="string", description="x")},
            handler=lambda x: x,
        )]

EXTENSION = _E()
"""
        _write_ext(tmp_path, "tool_ext.py", code)
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg = _make_registry(tmp_path)
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        assert "my_tool" in r.names()

    def test_tools_callable_via_registry_dispatch(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest, ToolDefinition, ToolParameter

class _E(Extension):
    manifest = ExtensionManifest(name="dispatch_ext")
    def tools(self):
        return [ToolDefinition(
            name="echo", description="d",
            parameters={"msg": ToolParameter(type="string", description="m")},
            handler=lambda msg: f"echo:{msg}",
        )]

EXTENSION = _E()
"""
        _write_ext(tmp_path, "dispatch_ext.py", code)
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg = _make_registry(tmp_path)
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        from tau.core.types import ToolCall
        result = r.dispatch(ToolCall(id="1", name="echo", arguments={"msg": "hi"}))
        assert result.content == "echo:hi"
        assert not result.is_error

    def test_tools_raising_exception_skipped_gracefully(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="bad_tools")
    def tools(self):
        raise RuntimeError("tools() exploded")

EXTENSION = _E()
"""
        _write_ext(tmp_path, "bad_tools.py", code)
        reg = _make_registry(tmp_path)
        loaded = _load(reg)
        # Extension is still "registered" (name discovered) but tools() error is absorbed
        assert "bad_tools" in loaded


# ===========================================================================
# ExtensionRegistry — system prompt fragment
# ===========================================================================

class TestExtensionRegistryPromptFragment:
    def test_prompt_fragment_injected_into_context(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(
        name="frag_ext",
        system_prompt_fragment="You also know how to juggle.",
    )

EXTENSION = _E()
"""
        _write_ext(tmp_path, "frag_ext.py", code)
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg = _make_registry(tmp_path)
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        sys_msgs = [m for m in c.get_messages() if m.role == "system"]
        combined = " ".join(m.content for m in sys_msgs)
        assert "juggle" in combined

    def test_no_fragment_no_injection(self, tmp_path):
        _write_ext(tmp_path, "no_frag.py", _minimal_ext("no_frag"))
        r = ToolRegistry()
        c = ContextManager(_cfg())
        base_sys = [m.content for m in c.get_messages() if m.role == "system"]
        reg = _make_registry(tmp_path)
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        new_sys = [m.content for m in c.get_messages() if m.role == "system"]
        assert base_sys == new_sys


# ===========================================================================
# ExtensionRegistry — slash commands
# ===========================================================================

class TestExtensionRegistrySlashCommands:
    def _build_ext_with_slash(self, tmp_path: Path, name: str, cmd: str, response: str) -> ExtensionRegistry:
        code = f"""\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand

class _E(Extension):
    manifest = ExtensionManifest(name={name!r})
    def slash_commands(self):
        return [SlashCommand(name={cmd!r}, description="test cmd")]
    def handle_slash(self, command, args, ctx):
        if command == {cmd!r}:
            ctx.print({response!r})
            return True
        return False

EXTENSION = _E()
"""
        _write_ext(tmp_path, f"{name}.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        return reg

    def test_slash_command_registered(self, tmp_path):
        reg = self._build_ext_with_slash(tmp_path, "scmd_ext", "foo", "bar")
        cmds = dict(reg.all_slash_commands())
        assert "foo" in cmds

    def test_handle_slash_dispatches_to_extension(self, tmp_path):
        reg = self._build_ext_with_slash(tmp_path, "dispatch_slash", "greet", "hello!")
        prints: list[str] = []
        ec = ExtensionContext(
            registry=ToolRegistry(),
            context=ContextManager(_cfg()),
            steering=None,
            console_print=prints.append,
        )
        handled = reg.handle_slash("/greet world", ec)
        assert handled is True
        assert "hello!" in prints

    def test_handle_slash_returns_false_for_unknown(self, tmp_path):
        reg = _make_registry(tmp_path)
        _load(reg)
        ec = ExtensionContext(
            registry=ToolRegistry(), context=ContextManager(_cfg()),
            steering=None, console_print=lambda _: None,
        )
        assert reg.handle_slash("/unknown cmd", ec) is False

    def test_handle_slash_passes_args(self, tmp_path):
        code = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand

class _E(Extension):
    manifest = ExtensionManifest(name="args_ext")
    def slash_commands(self): return [SlashCommand(name="echo")]
    def handle_slash(self, command, args, ctx):
        ctx.print(f"got:{args}")
        return True

EXTENSION = _E()
"""
        _write_ext(tmp_path, "args_ext.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        prints: list[str] = []
        ec = ExtensionContext(
            registry=ToolRegistry(), context=ContextManager(_cfg()),
            steering=None, console_print=prints.append,
        )
        reg.handle_slash("/echo hello world", ec)
        assert "got:hello world" in prints

    def test_duplicate_slash_command_overwritten(self, tmp_path):
        code_a = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand
class _E(Extension):
    manifest = ExtensionManifest(name="ext_a")
    def slash_commands(self): return [SlashCommand(name="clash")]
    def handle_slash(self, command, args, ctx):
        ctx.print("from_a"); return True
EXTENSION = _E()
"""
        code_b = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand
class _E(Extension):
    manifest = ExtensionManifest(name="ext_b")
    def slash_commands(self): return [SlashCommand(name="clash")]
    def handle_slash(self, command, args, ctx):
        ctx.print("from_b"); return True
EXTENSION = _E()
"""
        _write_ext(tmp_path, "ext_a.py", code_a)
        _write_ext(tmp_path, "ext_b.py", code_b)
        reg = _make_registry(tmp_path)
        _load(reg)
        prints: list[str] = []
        ec = ExtensionContext(
            registry=ToolRegistry(), context=ContextManager(_cfg()),
            steering=None, console_print=prints.append,
        )
        reg.handle_slash("/clash", ec)
        # Exactly one handler should have fired
        assert len(prints) == 1

    def test_all_slash_commands_sorted(self, tmp_path):
        for name, cmd in [("z_ext", "zzz"), ("a_ext", "aaa"), ("m_ext", "mmm")]:
            code = f"""\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest, SlashCommand
class _E(Extension):
    manifest = ExtensionManifest(name={name!r})
    def slash_commands(self): return [SlashCommand(name={cmd!r})]
EXTENSION = _E()
"""
            _write_ext(tmp_path, f"{name}.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        names = [c for c, _ in reg.all_slash_commands()]
        assert names == sorted(names)

    def test_handle_slash_exception_absorbed(self, tmp_path):
        code = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand
class _E(Extension):
    manifest = ExtensionManifest(name="boom_slash")
    def slash_commands(self): return [SlashCommand(name="boom")]
    def handle_slash(self, command, args, ctx):
        raise RuntimeError("exploded")
EXTENSION = _E()
"""
        _write_ext(tmp_path, "boom_slash.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        ec = ExtensionContext(
            registry=ToolRegistry(), context=ContextManager(_cfg()),
            steering=None, console_print=lambda _: None,
        )
        # Should not raise
        result = reg.handle_slash("/boom", ec)
        assert result is False


# ===========================================================================
# ExtensionRegistry — event hooks
# ===========================================================================

class TestExtensionRegistryEventHooks:
    def test_event_hook_called_for_turn_complete(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest, TurnComplete

class _E(Extension):
    manifest = ExtensionManifest(name="hook_ext")
    def __init__(self):
        self.fired = []
    def event_hook(self, event):
        self.fired.append(type(event).__name__)

EXTENSION = _E()
"""
        _write_ext(tmp_path, "hook_ext.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        reg.fire_hooks(TurnComplete(usage=TokenUsage(input_tokens=1, output_tokens=1)))
        ext = reg.get("hook_ext")
        assert "TurnComplete" in ext.fired

    def test_event_hook_not_registered_for_base_class(self, tmp_path):
        _write_ext(tmp_path, "no_hook.py", _minimal_ext("no_hook"))
        reg = _make_registry(tmp_path)
        _load(reg)
        # Base class does not override event_hook → _hooks list should be empty
        assert reg._hooks == []

    def test_event_hook_exception_absorbed(self, tmp_path):
        code = """\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="exc_hook")
    def event_hook(self, event):
        raise RuntimeError("hook exploded")

EXTENSION = _E()
"""
        _write_ext(tmp_path, "exc_hook.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        # Should not raise
        reg.fire_hooks(TurnComplete(usage=TokenUsage()))

    def test_multiple_hooks_all_called(self, tmp_path):
        for i in range(3):
            code = f"""\
from tau.core.extension import Extension
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="multi_hook_{i}")
    def __init__(self):
        self.count = 0
    def event_hook(self, event):
        self.count += 1

EXTENSION = _E()
"""
            _write_ext(tmp_path, f"multi_hook_{i}.py", code)

        reg = _make_registry(tmp_path)
        _load(reg)
        reg.fire_hooks(TurnComplete(usage=TokenUsage()))
        for i in range(3):
            ext = reg.get(f"multi_hook_{i}")
            assert ext.count == 1

    def test_fire_hooks_with_no_hooks_registered(self, tmp_path):
        reg = _make_registry(tmp_path)
        _load(reg)
        # Should not raise
        reg.fire_hooks(TurnComplete(usage=TokenUsage()))


# ===========================================================================
# ExtensionRegistry — on_load callback
# ===========================================================================

class TestExtensionOnLoad:
    def test_on_load_called_after_registration(self, tmp_path):
        code = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="on_load_ext")
    def __init__(self):
        self.loaded = False
    def on_load(self, ctx):
        self.loaded = True

EXTENSION = _E()
"""
        _write_ext(tmp_path, "on_load_ext.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        assert reg.get("on_load_ext").loaded is True

    def test_on_load_receives_extension_context(self, tmp_path):
        code = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="ctx_ext")
    def __init__(self):
        self.got_ctx = None
    def on_load(self, ctx):
        self.got_ctx = ctx

EXTENSION = _E()
"""
        _write_ext(tmp_path, "ctx_ext.py", code)
        reg = _make_registry(tmp_path)
        _load(reg)
        assert isinstance(reg.get("ctx_ext").got_ctx, ExtensionContext)

    def test_on_load_exception_absorbed(self, tmp_path):
        code = """\
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest

class _E(Extension):
    manifest = ExtensionManifest(name="load_boom")
    def on_load(self, ctx):
        raise RuntimeError("on_load exploded")

EXTENSION = _E()
"""
        _write_ext(tmp_path, "load_boom.py", code)
        reg = _make_registry(tmp_path)
        loaded = _load(reg)   # should not raise
        assert "load_boom" in loaded


# ===========================================================================
# ExtensionRegistry — introspection
# ===========================================================================

class TestExtensionRegistryIntrospection:
    def test_loaded_extensions_returns_manifests(self, tmp_path):
        _write_ext(tmp_path, "intro_ext.py", _minimal_ext("intro_ext"))
        reg = _make_registry(tmp_path)
        _load(reg)
        manifests = reg.loaded_extensions()
        assert any(m.name == "intro_ext" for m in manifests)

    def test_get_returns_extension_instance(self, tmp_path):
        _write_ext(tmp_path, "get_test.py", _minimal_ext("get_test"))
        reg = _make_registry(tmp_path)
        _load(reg)
        ext = reg.get("get_test")
        assert isinstance(ext, Extension)
        assert ext.manifest.name == "get_test"

    def test_get_returns_none_for_unknown(self, tmp_path):
        reg = _make_registry(tmp_path)
        _load(reg)
        assert reg.get("does_not_exist") is None

    def test_len_zero_when_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        _load(reg)
        assert len(reg) == 0

    def test_len_correct_after_loading(self, tmp_path):
        _write_ext(tmp_path, "a.py", _minimal_ext("a"))
        _write_ext(tmp_path, "b.py", _minimal_ext("b"))
        reg = _make_registry(tmp_path)
        _load(reg)
        assert len(reg) == 2


# ===========================================================================
# Built-in extensions (word_count + pretty_json)
# ===========================================================================

class TestBuiltinWordCount:
    def _load_builtin(self) -> tuple[ExtensionRegistry, ToolRegistry, ContextManager]:
        from pathlib import Path
        builtin_dir = Path(__file__).parent.parent / "tau" / "extensions"
        reg = ExtensionRegistry(extra_paths=[], disabled=[])
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        return reg, r, c

    def test_word_count_extension_loaded(self):
        reg, _, _ = self._load_builtin()
        assert reg.get("word_count") is not None

    def test_word_count_tool_registered(self):
        _, r, _ = self._load_builtin()
        assert "word_count" in r.names()

    def test_word_count_tool_correct_result(self):
        _, r, _ = self._load_builtin()
        from tau.core.types import ToolCall
        result = r.dispatch(ToolCall(id="1", name="word_count", arguments={"text": "hello world\nfoo"}))
        assert "words=3" in result.content
        assert "lines=2" in result.content

    def test_wc_slash_command_registered(self):
        reg, _, _ = self._load_builtin()
        cmds = dict(reg.all_slash_commands())
        assert "wc" in cmds

    def test_wc_slash_handles_text(self):
        reg, r, c = self._load_builtin()
        prints: list[str] = []
        ec = ExtensionContext(
            registry=r, context=c, steering=None, console_print=prints.append,
        )
        handled = reg.handle_slash("/wc hello world", ec)
        assert handled is True
        assert any("words=2" in p for p in prints)

    def test_wc_slash_no_args_prints_usage(self):
        reg, r, c = self._load_builtin()
        prints: list[str] = []
        ec = ExtensionContext(registry=r, context=c, steering=None, console_print=prints.append)
        reg.handle_slash("/wc", ec)
        assert any("Usage" in p or "usage" in p for p in prints)


class TestBuiltinPrettyJson:
    def _load_builtin(self) -> tuple[ExtensionRegistry, ToolRegistry, ContextManager]:
        reg = ExtensionRegistry(extra_paths=[], disabled=[])
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        return reg, r, c

    def test_pretty_json_extension_loaded(self):
        reg, _, _ = self._load_builtin()
        assert reg.get("pretty_json") is not None

    def test_pretty_print_json_tool_registered(self):
        _, r, _ = self._load_builtin()
        assert "pretty_print_json" in r.names()

    def test_pretty_print_json_formats_correctly(self):
        _, r, _ = self._load_builtin()
        from tau.core.types import ToolCall
        result = r.dispatch(ToolCall(
            id="1", name="pretty_print_json",
            arguments={"json_string": '{"b":2,"a":1}'},
        ))
        parsed = json.loads(result.content)
        assert parsed == {"b": 2, "a": 1}

    def test_pretty_print_json_invalid_input(self):
        _, r, _ = self._load_builtin()
        from tau.core.types import ToolCall
        result = r.dispatch(ToolCall(
            id="1", name="pretty_print_json",
            arguments={"json_string": "not json"},
        ))
        assert "Error" in result.content

    def test_json_slash_command_registered(self):
        reg, _, _ = self._load_builtin()
        cmds = dict(reg.all_slash_commands())
        assert "json" in cmds

    def test_json_slash_formats_inline(self):
        reg, r, c = self._load_builtin()
        prints: list[str] = []
        ec = ExtensionContext(registry=r, context=c, steering=None, console_print=prints.append)
        reg.handle_slash('/json {"x":1}', ec)
        combined = " ".join(prints)
        assert '"x"' in combined

    def test_event_hook_accumulates_tokens(self):
        reg, _, _ = self._load_builtin()
        ext = reg.get("pretty_json")
        assert ext.total_tokens == 0
        reg.fire_hooks(TurnComplete(usage=TokenUsage(input_tokens=10, output_tokens=5)))
        assert ext.total_tokens == 15
        reg.fire_hooks(TurnComplete(usage=TokenUsage(input_tokens=20, output_tokens=10)))
        assert ext.total_tokens == 45


# ===========================================================================
# ExtensionLoadError type
# ===========================================================================

class TestExtensionLoadError:
    def test_fields(self):
        e = ExtensionLoadError(extension_name="bad_ext", error="boom")
        assert e.extension_name == "bad_ext"
        assert e.error == "boom"
