# tau — Design Document
> A minimal, extensible CLI coding agent with multi-provider LLM support and tool use.

> Note (May 2026): this document contains historical sections from earlier milestones.
> See **Section 0: Current Architecture Snapshot** for the up-to-date shipped design.

---

## 0. Current Architecture Snapshot (May 2026)

### 0.1 Core Principles (unchanged)
- Minimal core runtime
- Extension-first capability growth
- Provider/tool abstractions with explicit policy boundaries

### 0.2 Current Provider Set
- OpenAI
- Anthropic
- Google Gemini
- Ollama
- MLX
- Unsloth

### 0.3 Core Additions Since Initial Draft
- Built-in task lifecycle extension (`task_create`, `task_update`, `task_list`) + plan mode
- Merkle code index module (`tau/core/code_index.py`) with:
  - change detection
  - persisted manifest/stats under `.tau/index/`
  - `/code-index-status` and `/code-index-refresh`
- MCP minimal built-in extension (`mcp_resources`) with:
  - `mcp_list_resources`, `mcp_read_resource`
  - `mcp_list_tools`, `mcp_call_tool`
  - slash commands `/mcp-resources`, `/mcp-tools`
- Prompt path integration for optional code-index context injection

### 0.4 Design Positioning
- Keep high-churn capabilities in extensions.
- Promote features into core only when they become always-on infrastructure
  (e.g., provider adapters, session DB, policy/runtime guardrails).

---

## 1. Goals & Philosophy

| Principle | Description |
|-----------|-------------|
| **Minimal core** | The core does one thing: run an agent loop (LLM ↔ tools). No bloat. |
| **Provider-agnostic** | Swap Google, OpenAI, Ollama, MLX, or any future provider via a unified interface. |
| **Tool-first** | The agent reasons by calling tools. Tools are first-class, declarative, and sandboxed. |
| **Extensible** | Skills (YAML + tools.py) and the new Extension system let anyone add tools, slash commands, event hooks, and prompt fragments at runtime. |
| **Session-aware** | Conversations have identity, history, branching, and can be resumed. |
| **CLI-native** | Designed for the terminal; no GUI, no server required. |
| **Resilient** | Auto-compaction, auto-retry with exponential backoff, and mid-stream steering make the agent robust in real use. |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                            tau CLI                               │
│  (Click entry point · REPL · single-shot · slash commands)       │
└──────────────────────┬───────────────────────────────────────────┘
                       │  Events (stream)
          ┌────────────▼────────────┐
          │        Agent Core       │
          │  ┌─────────────────┐    │
          │  │   agent loop    │    │
          │  │  + retry        │    │
          │  │  + compaction   │    │
          │  │  + steering     │    │
          │  └────────┬────────┘    │
          │           │             │
          │  ┌────────▼────────┐    │
          │  │ ContextManager  │    │
          │  │ (trim/compact)  │    │
          │  └─────────────────┘    │
          │  ┌─────────────────┐    │
          │  │ SessionManager  │    │
          │  │ (persist/fork)   │    │
          │  └─────────────────┘    │
          └──┬──────────────────────┘
             │
   ┌──────────▼──────┐   ┌─────────────────┐   ┌─────────────────┐
   │ Provider Layer  │   │  Tool Registry  │   │ExtensionRegistry│
   │ (LLM adapters)  │   │ built-in tools  │   │ tools           │
   │                 │   │ + skill tools   │   │ slash commands  │
   │ openai          │   │ + ext tools     │   │ event hooks     │
   │ google          │   │                 │   │ prompt frags    │
   │ ollama          │   └─────────────────┘   └─────────────────┘
   │ mlx             │           │                      │
   └─────────────────┘           │                      │
                        ┌────────▼──────────────────────▼────────┐
                        │         Skill Loader  (legacy)          │
                        │  discovers skills/ dirs at startup      │
                        └─────────────────────────────────────────┘
```

---

## 3. Directory Layout

```
tau/
├── tau/
│   ├── __init__.py
│   ├── cli.py                   # Click entry point, REPL, renderer
│   ├── config.py                # config loading (~/.tau/config.toml)
│   │
│   ├── core/
│   │   ├── agent.py             # agent loop (retry, compaction, steering)
│   │   ├── context.py           # token window, trim strategies, Compactor
│   │   ├── extension.py         # Extension base, ExtensionContext, ExtensionRegistry
│   │   ├── session.py           # session persistence, fork/branch
│   │   ├── steering.py          # SteeringChannel (steer + follow-up queue)
│   │   ├── tool_registry.py     # registration + dispatch
│   │   └── types.py             # all shared dataclasses / events
│   │
│   ├── providers/
│   │   ├── base.py              # BaseProvider protocol
│   │   ├── openai_provider.py
│   │   ├── google_provider.py
│   │   ├── ollama_provider.py
│   │   └── mlx_provider.py      # Apple Silicon local via mlx-lm
│   │
│   ├── tools/
│   │   ├── fs.py                # read/write/edit/list/search
│   │   └── shell.py             # run_bash
│   │
│   ├── skills/                  # legacy skill system (YAML + tools.py)
│   │   ├── __init__.py          # SkillLoader
│   │   └── example_skill/
│   │
│   ├── editor.py                # @file expansion, tab completion, image paste, !shell
│   ├── prompts.py               # prompt template discovery + rendering
│   ├── context_files.py         # AGENTS.md / CLAUDE.md / .tau/SYSTEM.md loading
│   ├── sdk.py                   # programmatic API (TauSession, create_session, InMemorySessionManager)
│   ├── rpc.py                   # JSONL-over-stdio RPC protocol for process integration
│   │
│   └── extensions/              # built-in extension system
│       ├── __init__.py
│       ├── word_count.py        # reference: tool + /wc slash command
│       └── pretty_json.py       # reference: tool + /json + event hook
│
├── tests/
│   ├── test_agent.py
│   ├── test_compaction.py
│   ├── test_context.py
│   ├── test_extensions.py       # 72 tests for the extension system
│   ├── test_fs_tools.py
│   ├── test_retry.py
│   ├── test_session.py
│   └── test_steering.py
│
├── pyproject.toml
├── README.md
└── DESIGN.md
```

User extension directories (searched in order):
```
tau/extensions/          ← built-ins (shipped with tau)
~/.tau/extensions/       ← user-installed
[extensions] paths = []  ← extra paths from config.toml
```

---

## 4. Module Breakdown

### 4.1 `core/types.py` — Shared Types

All types are plain dataclasses — no framework coupling.

**Messages & tools**
```
Message            role, content, tool_call_id, tool_calls, name
ToolDefinition     name, description, parameters (JSON Schema), handler fn
ToolCall           id, name, arguments
ToolResult         tool_call_id, content, is_error
ToolParameter      type, description, enum, required
```

**Agent config**
```
AgentConfig        provider, model, max_tokens, max_turns, system_prompt,
                   trim_strategy, workspace_root,
                   compaction_enabled, compaction_threshold,
                   retry_enabled, retry_max_attempts, retry_base_delay,
                   parallel_tools, parallel_tools_max_workers,
                   max_cost (USD ceiling; 0 = unlimited)
```

**Extension types**
```
ExtensionManifest  name, version, description, author, system_prompt_fragment
SlashCommand       name, description, usage
ExtensionLoadError extension_name, error   (emitted as an Event)
```

**Event stream** — everything yielded by `agent.run()`:
```
TextDelta          streaming token (is_thinking flag for CoT models)
TextChunk          complete assistant text block
ToolCallEvent      a tool the LLM wants to call
ToolResultEvent    the result returned to the LLM
TurnComplete       end of one turn + TokenUsage
CompactionEvent    stage=start|end, token counts, summary
RetryEvent         attempt, max_attempts, delay, error
SteerEvent         new_input, discarded_tokens
CostLimitExceeded  session_cost, max_cost (budget exceeded)
ExtensionLoadError extension_name, error
ErrorEvent         message
```

---

### 4.2 `core/context.py` — Context Manager + Compactor

**ContextManager** responsibilities:
- Maintain the **message list**.
- Enforce a **token budget** (sliding window or summarisation).
- Inject system prompt + skill/extension prompt fragments.
- `inject_prompt_fragment()` appends to the system message.

**Compactor** responsibilities:
- `should_compact(messages)` — true when token usage ≥ threshold × max_tokens.
- `compact(messages, provider)` — calls the LLM for a rolling summary, keeps the last N messages verbatim, returns a trimmed list + `CompactionEntry`.
- `is_overflow_error(msg)` — detects context-length errors for overflow recovery.

Trimming strategies:

| Strategy | Behaviour |
|----------|-----------|
| `sliding_window` | Drop oldest non-system messages when over budget |
| `summarise` | Ask the LLM to summarise old messages, replace with summary |

---

### 4.3 `core/session.py` — Session Manager

Responsibilities:
- UUID + timestamp + optional name per session.
- Persist to `~/.tau/sessions/<id>.json`.
- Resume by ID prefix.
- **Fork** a session at any message index → new session, parent_id + fork_index recorded.
- **List branches** (direct children of a session).
- **Get fork points** (all user messages eligible as fork targets).
- Append `CompactionEntry` records to a session.

Session file schema:
```json
{
  "id": "uuid",
  "name": "optional name",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "parent_id": "uuid | null",
  "fork_index": 3,
  "config": { "provider": "openai", "model": "gpt-4o", ... },
  "messages": [ ... ],
  "compactions": [ { "summary": "...", "tokens_before": 4200, "timestamp": "..." } ]
}
```

---

### 4.4 `core/tool_registry.py` — Tool Registry & Dispatcher

- Central `name → ToolDefinition` map.
- `register()` / `register_many()` — called by built-ins, SkillLoader, ExtensionRegistry.
- `dispatch(call)` — invokes handler, wraps exceptions in `ToolResult(is_error=True)`.
- `all_definitions()` — used by providers to build their tool schema.

---

### 4.5 `core/agent.py` — Agent Loop

ReAct-style loop with three resilience layers:

```
agent.run(user_input):

  [retry loop — up to retry_max_attempts]
    1. context.trim()
    2. if compactor.should_compact(): run compaction → emit CompactionEvents
    3. response = provider.chat(messages, tools)   ← streaming or blocking
       └─ if streaming: check SteeringChannel after each TextDelta
          └─ if steer arrives: emit SteerEvent, restart turn with new input
    4. if response.tool_calls:
         for call in tool_calls:
           result = registry.dispatch(call)
           emit ToolCallEvent, ToolResultEvent
           context.add_message(tool_result)
    5. else: emit TextChunk/TextDelta, TurnComplete; break
    6. if provider raises retryable error: emit RetryEvent, sleep, retry
    7. if provider raises overflow error: compact, retry once

  [follow-up queue — after each TurnComplete]
    if steering.dequeue(): loop back with next queued prompt
```

**Retry policy** (configurable via `AgentConfig`):
- Exponential backoff: `base_delay × 2^(attempt-1)`.
- Retryable: rate limits, 5xx errors, network timeouts.
- Non-retryable: auth errors, bad requests, overflow (handled by compaction).

---

### 4.6 `core/steering.py` — SteeringChannel

Thread-safe communication between the REPL input thread and the agent loop.

| Mechanism | Write (REPL thread) | Read (agent thread) | Semantics |
|-----------|--------------------|--------------------|-----------|
| **Steer** | `steer(msg)` | `consume_steer()` after each delta | Interrupt current stream, restart turn |
| **Queue** | `enqueue(msg)` | `dequeue()` after TurnComplete | FIFO follow-up prompts |

---

### 4.7 `core/extension.py` — Extension System

The extension system is the primary extensibility mechanism for tau. It supersedes the older skills system for new work (skills remain supported for backwards compatibility).

#### `Extension` base class

```python
class Extension:
    manifest: ExtensionManifest          # MUST set as class attribute

    def tools(self) -> list[ToolDefinition]: ...          # register tools
    def slash_commands(self) -> list[SlashCommand]: ...   # register /commands
    def handle_slash(self, cmd, args, ctx) -> bool: ...   # handle /command
    def event_hook(self, event: Event) -> None: ...       # observe all events
    def on_load(self, ctx: ExtensionContext) -> None: ... # startup callback
    def on_unload(self) -> None: ...                      # teardown (future)
```

Every extension file must expose a module-level `EXTENSION` instance.

#### `ExtensionContext` façade

Passed to `on_load()` and `handle_slash()`. Gives extensions safe access to:
- `register_tool(tool)` — register additional tools at runtime.
- `registered_tools()` — list of currently registered tool names.
- `enqueue(msg)` — add a follow-up prompt to the steering queue.
- `print(text)` — write to the REPL console (Rich markup supported).
- `token_count()` — current context token usage.

#### `ExtensionRegistry`

Discovery and lifecycle management:

```python
ExtensionRegistry(
    extra_paths=[],       # additional search dirs
    disabled=[],          # extension names to skip
    include_builtins=True # set False in tests for isolation
)
```

Search order (first match by name wins):
1. `tau/extensions/` (built-ins)
2. `~/.tau/extensions/` (user)
3. Extra paths from `config.toml → [extensions] paths`

Supported file layouts:
```
my_ext.py              ← single-file extension
my_ext/extension.py    ← package (preferred entry point)
my_ext/__init__.py     ← package (fallback)
```

All load/dispatch/hook errors are caught and logged — a bad extension never crashes tau.

Runtime API:
- `load_all(registry, context, steering, console_print)` — discover + register all.
- `handle_slash(raw_input, ext_context)` — route `/command` to owning extension.
- `fire_hooks(event)` — broadcast an event to all registered hooks.
- `loaded_extensions()` → `list[ExtensionManifest]`
- `all_slash_commands()` → sorted `list[(name, description)]`
- `get(name)` → `Extension | None`

---

### 4.8 `providers/base.py` — Provider Protocol

```python
class BaseProvider(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse | Generator[TextDelta | ProviderResponse, None, None]: ...
```

`ProviderResponse`:
```
content       str | None
tool_calls    list[ToolCall]
stop_reason   "end_turn" | "tool_use" | "max_tokens" | "error"
usage         TokenUsage(input_tokens, output_tokens)
```

---

### 4.9 `tools/fs.py` — Filesystem Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file, optionally slice by line range |
| `write_file` | Create or overwrite a file |
| `edit_file` | Targeted patch (old_str → new_str) |
| `list_dir` | List directory contents |
| `search_files` | Grep / glob across the workspace (supports changed-only filtering) |

All paths validated against a configurable **workspace root**.

---

### 4.10 `tools/shell.py` — Shell Tool

| Tool | Description |
|------|-------------|
| `run_bash` | Run shell command, capture stdout/stderr/exit-code |

Safety flags: `require_confirmation`, `allowed_commands`, `timeout`.

---

### 4.11 `skills/` — Legacy Skill System

A **skill** is a directory containing:
- `skill.yaml` — name, version, description, optional system prompt fragment.
- `tools.py` — `TOOLS: list[ToolDefinition]`.

`SkillLoader` scans built-in `tau/skills/`, `~/.tau/skills/`, and extra paths from config. Fully backwards-compatible; new extensibility work should use the Extension system instead.

---

## 5. CLI Design (`cli.py`)

```
tau run                              # interactive REPL
tau run "fix the bug in foo.py"      # single-shot
tau run -p ollama -m llama3 "..."    # choose provider/model
tau run -s <session-id>              # resume session

tau sessions list
tau sessions show <id>
tau sessions delete <id>
tau sessions fork <id> <index>       # branch at message index
tau sessions fork <id> <index> --resume
tau sessions branches <id>           # list direct forks
tau sessions fork-points <id>        # list forkable user messages

tau extensions list                  # list loaded extensions
tau extensions show <name>           # detail view for one extension

tau config show
tau config set <key> <value>
```

**REPL slash commands** (built-in):

| Command | Description |
|---------|-------------|
| `/queue <msg>` | Add follow-up prompt to the queue |
| `/queue` | Show queue size |
| `/steer <msg>` | Inject mid-stream steer |
| `/clear` | Cancel pending steer |
| `/help` | Show all commands (built-in + extension) |
| `exit` / Ctrl-D | Quit |

Extension slash commands are auto-listed in `/help` and routed via `ExtensionRegistry.handle_slash()`.

**Built-in extension slash commands:**

| Command | Extension | Description |
|---------|-----------|-------------|
| `/wc <text>` | word_count | Count words/lines/chars |
| `/json <json>` | pretty_json | Pretty-print JSON inline |

---

### 5.1 `editor.py` — Editor Richness

The `editor` module provides four REPL enhancements:

| Feature | Trigger | Scope |
|---------|---------|-------|
| **@file references** | `@path/to/file` in prompt text | REPL + single-shot |
| **Tab completion** | `Tab` key in REPL | REPL only |
| **Image paste** | `Ctrl-V` in REPL (macOS) | REPL only |
| **Inline shell** | `!command` in REPL | REPL only |

**@file expansion** — `expand_at_files(text, workspace_root)`:
- Regex `@path` matches against real files inside the workspace.
- Matched files are read and replaced with `<file path="...">contents</file>` blocks.
- Files outside the workspace are blocked (left as-is). Files > 256 KB are skipped.
- Non-existent `@references` pass through unchanged so the LLM can still interpret them.

**Tab completer** — `_TauCompleter` (prompt_toolkit `Completer`):
- `/` prefix → slash command completion (built-in + extension commands).
- `@` prefix → filesystem path completion within the workspace.
- Hidden (dotfiles) only shown when the typed prefix starts with `.`.

**Image paste** — `Ctrl-V` handler:
- On macOS, uses AppleScript to extract clipboard PNG data to a temp file.
- Falls back to normal paste if no image is in the clipboard.
- Stages the image via `_staged_images`, same as `/image <path>`.

**Inline shell** — `!command`:
- Runs via `subprocess.run()` in the workspace directory.
- Output displayed directly in the REPL, bypasses the agent.
- 30-second timeout by default.

---

## 6. Configuration (`config.py`)

Config file: `~/.tau/config.toml`

```toml
[defaults]
provider      = "openai"
model         = "gpt-4o"
max_tokens    = 8192
max_turns     = 20
trim_strategy = "sliding_window"

[providers.openai]
api_key  = "sk-..."          # or env: OPENAI_API_KEY
base_url = "https://api.openai.com/v1"

[providers.google]
api_key = "..."              # or env: GOOGLE_API_KEY

[providers.ollama]
base_url = "http://localhost:11434"

[tools.shell]
require_confirmation = true
timeout = 30
allowed_commands = []        # empty = allow all

[skills]
paths    = ["~/my-skills"]
disabled = []

[extensions]
paths    = ["~/my-extensions"]
disabled = ["word_count"]    # opt out of a built-in

[theme]
user_color      = "cyan"     # customise any Rich style name
assistant_color = "green"
tool_color      = "yellow"
accent_color    = "cyan"
border_style    = "dim"

[tools]
disabled     = ["run_bash"]  # disable specific built-in tools
# enabled_only = ["read_file", "list_dir"]  # OR: whitelist only these
```

Environment variables always override config file values (`TAU_PROVIDER`, `TAU_MODEL`, `TAU_EXTENSIONS_DISABLED`, etc.).

---

## 7. Data Flow (Single Turn)

```
User input
    │
    ▼
cli.py ──► agent.run(input)
                │
                ├─ context.add_message(user_msg)
                ├─ context.trim()
                ├─ compactor.should_compact()? ──► compact() ──► CompactionEvent
                │
                ├─ provider.chat(messages, tools)   ◄── LLM API
                │       │
                │    streaming?
                │    ├─ yes: yield TextDelta per token
                │    │       check SteeringChannel after each delta
                │    │       steer? ──► SteerEvent ──► restart turn
                │    └─ no:  blocking ProviderResponse
                │
                │   ProviderResponse
                │    ├─ tool_calls ──► registry.dispatch()
                │    │                ──► yield ToolCallEvent, ToolResultEvent
                │    │                ──► context.add_message(tool_result)
                │    │                ──► loop
                │    └─ final text ──► yield TextChunk / TurnComplete
                │                 ──► ext_registry.fire_hooks(TurnComplete)
                │                 ──► session.save()
                │
                ├─ retryable error? ──► yield RetryEvent ──► sleep ──► retry
                ├─ overflow error?  ──► compact ──► retry once
                │
                └─ steering.dequeue()? ──► loop with next queued prompt
```

---

## 8. Extension System — Writing an Extension

Minimal single-file extension (`~/.tau/extensions/my_ext.py`):

```python
from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand, ToolDefinition, ToolParameter

class MyExtension(Extension):
    manifest = ExtensionManifest(
        name="my_ext",
        version="1.0.0",
        description="Does something useful.",
        author="you",
        system_prompt_fragment="You also have my_ext tools available.",
    )

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="my_tool",
                description="Does something.",
                parameters={"x": ToolParameter(type="string", description="input")},
                handler=lambda x: f"result: {x}",
            )
        ]

    def slash_commands(self) -> list[SlashCommand]:
        return [SlashCommand(name="do", description="Run my tool.", usage="/do <input>")]

    def handle_slash(self, command: str, args: str, ctx: ExtensionContext) -> bool:
        if command != "do":
            return False
        ctx.print(f"result: {args}")
        return True

    def event_hook(self, event) -> None:
        from tau.core.types import TurnComplete
        if isinstance(event, TurnComplete):
            print(f"turn used {event.usage.total} tokens")

    def on_load(self, ctx: ExtensionContext) -> None:
        ctx.print("[dim]my_ext loaded[/dim]")

EXTENSION = MyExtension()
```

Package layout (for complex extensions):
```
~/.tau/extensions/my_ext/
    extension.py       ← preferred entry point
    helpers.py
    data/
```

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `click` | CLI arg parsing & REPL |
| `rich` | Terminal rendering |
| `openai` | OpenAI + compatible APIs |
| `anthropic` | Anthropic Claude |
| `google-genai` | Google Gemini |
| `httpx` | Ollama (plain HTTP) |
| `pydantic` + `pydantic-settings` | Config validation |
| `tomllib` (stdlib ≥ 3.11) / `tomli` | Config parsing |
| `pyyaml` | Skill YAML parsing |

No LangChain, no heavy framework. The core is hand-rolled and stays under ~2 kloc.

---

## 10. Implementation Status

| Feature | Status | Tests |
|---------|--------|-------|
| Core agent loop | ✅ done | `test_agent.py` |
| Provider layer (OpenAI / Google / Ollama) | ✅ done | — |
| Filesystem tools | ✅ done | `test_fs_tools.py` |
| Shell tool | ✅ done | — |
| Context manager (sliding window + summarise) | ✅ done | `test_context.py` |
| Session persistence & resume | ✅ done | `test_session.py` |
| **Auto-compaction** (threshold + overflow recovery) | ✅ done | `test_compaction.py` |
| **Auto-retry** with exponential backoff | ✅ done | `test_retry.py` |
| **Mid-stream steering** + follow-up queue | ✅ done | `test_steering.py` |
| **Session branching** (fork / branches / fork-points) | ✅ done | `test_session.py` |
| **Extension system** | ✅ done | `test_extensions.py` (72 tests) |
| Legacy skills system | ✅ done | — |
| `tau extensions list/show` CLI commands | ✅ done | — |
| **Output modes** (print / JSON / piped stdin) | ✅ done | `test_output_modes.py` |
| **System prompt override** (`.tau/SYSTEM.md`) | ✅ done | `test_context_files.py` |
| **Prompt templates** (`{{variables}}`) | ✅ done | `test_prompts.py` |
| **Editor richness** (@file, tab, image paste, !shell) | ✅ done | `test_editor.py` (40 tests) |
| **SDK** (`TauSession`, `create_session`) | ✅ done | `test_sdk.py` |
| **RPC** (JSONL-over-stdio protocol) | ✅ done | `test_rpc.py` |
| **/copy** slash command | ✅ done | `test_slash_commands.py` |
| **Session export** (JSON + Markdown) | ✅ done | `test_session.py`, `test_slash_commands.py` |
| **Themes** (configurable Rich colors) | ✅ done | `test_theme_and_tools_config.py` |
| **Configurable tool set** (disabled / enabled_only) | ✅ done | `test_theme_and_tools_config.py`, `test_tool_registry.py` |
| **/reload** hot-reload (config, extensions, skills) | ✅ done | `test_slash_commands.py`, `test_extensions.py` |
| **`--max-cost` budget guard** (CostLimitExceeded event) | ✅ done | `test_max_cost.py` |
| **`--no-session` ephemeral mode** (InMemorySessionManager) | ✅ done | `test_max_cost.py` |

**Total: 556 tests, all passing.**
