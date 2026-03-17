# tau — Design Document

> A minimal, extensible CLI coding agent with multi-provider LLM support and tool use.

---

## 1. Goals & Philosophy

| Principle | Description |
|-----------|-------------|
| **Minimal core** | The core does one thing: run an agent loop (LLM ↔ tools). No bloat. |
| **Provider-agnostic** | Swap Google, OpenAI, Ollama, or any future provider via a unified interface. |
| **Tool-first** | The agent reasons by calling tools. Tools are first-class, declarative, and sandboxed. |
| **Extensible via skills** | Skills are bundles of tools + optional prompt fragments, loaded at runtime. |
| **Session-aware** | Conversations have identity, history, and can be resumed. |
| **CLI-native** | Designed for the terminal; no GUI, no server required to run. |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                          tau CLI                            │
│  (entry point, arg parsing, REPL / single-shot mode)        │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         Agent Core          │
          │  - agent loop               │
          │  - context manager          │
          │  - session manager          │
          │  - tool dispatcher          │
          └──┬──────────────────┬───────┘
             │                  │
   ┌──────────▼──────┐   ┌──────▼──────────┐
   │  Provider Layer │   │   Tool Registry  │
   │  (LLM adapters) │   │  (built-in +     │
   │                 │   │   skill tools)   │
   │  - openai       │   │                  │
   │  - google       │   │  - fs (read/     │
   │  - ollama       │   │    write/edit)   │
   │  - base         │   │  - shell (bash)  │
   └─────────────────┘   │  - search        │
                         └──────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │       Skill Loader       │
                     │  (discovers & registers  │
                     │   skills at startup)     │
                     └─────────────────────────┘
```

---

## 3. Directory Layout

```
tau/
├── tau/                        # main package
│   ├── __init__.py
│   ├── cli.py                  # entry point (Click / argparse)
│   │
│   ├── core/                   # ── CORE MODULE ──────────────────
│   │   ├── __init__.py
│   │   ├── agent.py            # agent loop
│   │   ├── context.py          # context / token window management
│   │   ├── session.py          # session persistence & resume
│   │   ├── tool_registry.py    # tool registration & dispatch
│   │   └── types.py            # shared dataclasses / protocols
│   │
│   ├── providers/              # ── PROVIDER MODULE ──────────────
│   │   ├── __init__.py
│   │   ├── base.py             # BaseProvider protocol / ABC
│   │   ├── openai_provider.py
│   │   ├── google_provider.py
│   │   └── ollama_provider.py
│   │
│   ├── tools/                  # ── BUILT-IN TOOLS ───────────────
│   │   ├── __init__.py
│   │   ├── fs.py               # read_file, write_file, edit_file, list_dir
│   │   └── shell.py            # run_bash
│   │
│   ├── skills/                 # ── SKILLS (extensible) ──────────
│   │   ├── __init__.py         # skill loader
│   │   └── example_skill/
│   │       ├── __init__.py
│   │       ├── skill.yaml      # metadata + system prompt fragment
│   │       └── tools.py        # skill-specific tools
│   │
│   └── config.py               # config loading (~/.tau/config.toml)
│
├── tests/
├── pyproject.toml
├── README.md
└── DESIGN.md                   # this file
```

---

## 4. Module Breakdown

### 4.1 `core/types.py` — Shared Types

```
Message          role (system|user|assistant|tool), content, tool_call_id
ToolDefinition   name, description, parameters (JSON Schema), handler fn
ToolCall         id, name, arguments (dict)
ToolResult       tool_call_id, content, is_error
AgentConfig      provider, model, max_tokens, max_turns, system_prompt
```

All types are plain dataclasses or `TypedDict`s — no framework coupling.

---

### 4.2 `core/context.py` — Context Manager

Responsibilities:
- Maintain the **message list** for the current turn.
- Enforce a **token budget** (sliding window or summarisation strategy).
- Inject **system prompt** + skill prompt fragments.
- Provide `add_message()`, `get_messages()`, `trim()`, `snapshot()`.

Trimming strategies (pluggable):
| Strategy | Behaviour |
|----------|-----------|
| `sliding_window` | Drop oldest non-system messages when over budget |
| `summarise` | Ask the LLM to summarise old messages, replace with summary |

```python
class ContextManager:
    def __init__(self, config: AgentConfig): ...
    def add_message(self, msg: Message) -> None: ...
    def get_messages(self) -> list[Message]: ...
    def token_count(self) -> int: ...          # rough estimate
    def trim(self) -> None: ...                # apply trim strategy
    def snapshot(self) -> list[Message]: ...   # serialisable copy
```

---

### 4.3 `core/session.py` — Session Manager

Responsibilities:
- Give each session a **UUID + timestamp + optional name**.
- **Persist** sessions to `~/.tau/sessions/<id>.json`.
- **Resume** a session by ID (restores message history into ContextManager).
- List / delete sessions.

```
~/.tau/
├── config.toml
└── sessions/
    ├── <uuid>.json
    └── ...
```

Session file schema:
```json
{
  "id": "uuid",
  "name": "optional human name",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "config": { "provider": "openai", "model": "gpt-4o" },
  "messages": [ ... ]
}
```

```python
class SessionManager:
    def new_session(self, config: AgentConfig, name: str | None) -> Session: ...
    def save(self, session: Session, messages: list[Message]) -> None: ...
    def load(self, session_id: str) -> Session: ...
    def list_sessions(self) -> list[SessionMeta]: ...
    def delete(self, session_id: str) -> None: ...
```

---

### 4.4 `core/tool_registry.py` — Tool Registry & Dispatcher

Responsibilities:
- Central registry: `name → ToolDefinition`.
- **Register** built-in tools at startup; skills register additional tools.
- **Dispatch** tool calls from the LLM: validate args, invoke handler, return `ToolResult`.
- Provide tool schemas in the format each provider expects (OpenAI function-calling, Google `FunctionDeclaration`, etc.).

```python
class ToolRegistry:
    def register(self, tool: ToolDefinition) -> None: ...
    def get(self, name: str) -> ToolDefinition: ...
    def all_definitions(self) -> list[ToolDefinition]: ...
    def dispatch(self, call: ToolCall) -> ToolResult: ...
```

---

### 4.5 `core/agent.py` — Agent Loop

This is the heart of tau. It runs the **ReAct-style loop**:

```
while turns < max_turns:
    1. context.trim()                          # enforce token budget
    2. response = provider.chat(messages, tools)
    3. if response.is_final_answer:
           yield AssistantMessage; break
    4. for each tool_call in response.tool_calls:
           result = registry.dispatch(tool_call)
           context.add_message(tool_result_msg)
    5. context.add_message(assistant_msg)
    6. turns += 1
```

```python
class Agent:
    def __init__(self, config, provider, registry, context, session): ...
    def run(self, user_input: str) -> Generator[Event, None, None]: ...
    # Events: TextChunk | ToolCallEvent | ToolResultEvent | ErrorEvent
```

Streaming-first: the loop yields typed `Event` objects so the CLI can render incrementally.

---

### 4.6 `providers/base.py` — Provider Protocol

```python
class BaseProvider(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        stream: bool = True,
    ) -> ProviderResponse: ...

    @property
    def name(self) -> str: ...
```

`ProviderResponse` normalises the LLM response:
```
content       str | None
tool_calls    list[ToolCall]
stop_reason   "end_turn" | "tool_use" | "max_tokens" | "error"
usage         TokenUsage(input, output)
```

Each concrete provider (`openai_provider.py`, `google_provider.py`, `ollama_provider.py`) adapts the SDK response to this shape.

---

### 4.7 `tools/fs.py` — Filesystem Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read a file, optionally slice by line range |
| `write_file` | Create or overwrite a file |
| `edit_file` | Apply a targeted patch (old_str → new_str) |
| `list_dir` | List directory contents |
| `search_files` | Grep / glob across the workspace |

All paths are validated against a configurable **workspace root** to prevent escapes.

---

### 4.8 `tools/shell.py` — Shell Tool

| Tool | Description |
|------|-------------|
| `run_bash` | Run a shell command, capture stdout/stderr/exit-code |

Safety flags (configurable):
- `require_confirmation` — ask user `y/n` before executing.
- `allowed_commands` — allowlist of command prefixes.
- `timeout` — hard kill after N seconds.

---

### 4.9 `skills/` — Skill System

A **skill** is a directory containing:
- `skill.yaml` — name, version, description, optional system prompt fragment.
- `tools.py` — additional `ToolDefinition`s to register.

The `SkillLoader` scans:
1. Built-in `tau/skills/`
2. `~/.tau/skills/`
3. Any path in `config.toml → [skills] paths`

```python
class SkillLoader:
    def discover(self) -> list[Skill]: ...
    def load_into(self, registry: ToolRegistry, context: ContextManager) -> None: ...
```

This makes tau trivially extensible: drop a folder, restart, new tools available.

---

## 5. CLI Design (`cli.py`)

```
tau                          # interactive REPL (default)
tau "fix the bug in foo.py"  # single-shot mode
tau --provider google --model gemini-2.0-flash "..."
tau --session <id>           # resume a session
tau sessions list
tau sessions show <id>
tau sessions delete <id>
tau config set provider openai
tau config set model gpt-4o
```

The REPL renders:
- Assistant text with markdown (via `rich`).
- Tool calls as collapsible blocks: `▶ run_bash("pytest tests/")`.
- Tool results (stdout/stderr) in a dim style.
- Token usage per turn in the status bar.

---

## 6. Configuration (`config.py`)

Config file: `~/.tau/config.toml`

```toml
[defaults]
provider = "openai"
model    = "gpt-4o"
max_tokens = 8192
max_turns  = 20
trim_strategy = "sliding_window"

[providers.openai]
api_key = "sk-..."          # or env: OPENAI_API_KEY

[providers.google]
api_key = "..."             # or env: GOOGLE_API_KEY

[providers.ollama]
base_url = "http://localhost:11434"

[tools.shell]
require_confirmation = true
timeout = 30

[skills]
paths = ["~/my-skills"]
```

Environment variables always override config file values.

---

## 7. Data Flow (Single Turn)

```
User input
    │
    ▼
cli.py  ──► agent.run(input)
                │
                ├─ context.add_message(user_msg)
                ├─ context.trim()
                │
                ├─ provider.chat(messages, tools)  ◄── LLM API
                │       │
                │       ▼
                │   ProviderResponse
                │       │
                │  ┌────┴──────┐
                │  │ tool_calls│          │ final text │
                │  └────┬──────┘          └─────┬──────┘
                │       │                       │
                │  registry.dispatch()      yield TextChunk
                │       │                  session.save()
                │  ToolResult
                │       │
                │  context.add_message(tool_result)
                │       │
                └───────┘ (loop)
```

---

## 8. Extension Points (Future)

| Area | How to extend |
|------|---------------|
| New LLM provider | Implement `BaseProvider`, register in `providers/__init__.py` |
| New built-in tool | Add function to `tools/`, register in `tools/__init__.py` |
| New skill | Drop a folder in `~/.tau/skills/` |
| Context strategy | Implement `BaseTrimStrategy`, set in config |
| Output renderer | Swap the `Renderer` used in `cli.py` |
| Auth / secrets | Pluggable `SecretStore` in `config.py` |

---

## 9. Dependencies (Minimal)

| Package | Purpose |
|---------|---------|
| `click` | CLI arg parsing & REPL |
| `rich` | Terminal rendering |
| `openai` | OpenAI + compatible APIs |
| `google-genai` | Google Gemini |
| `httpx` | Ollama (plain HTTP) |
| `pydantic` | Config validation |
| `tomllib` (stdlib 3.11+) | Config parsing |

No LangChain, no heavy framework. Everything in tau is hand-rolled and < 1 kloc for the core.

---

## 10. Phased Implementation Plan

| Phase | Deliverable |
|-------|-------------|
| **P0** | `core/types`, `core/tool_registry`, `tools/fs`, `tools/shell`, `providers/base` + `openai` |
| **P1** | `core/agent` loop, `cli.py` REPL, `core/context` sliding-window |
| **P2** | `core/session` persistence & resume, `config.py` |
| **P3** | `providers/google`, `providers/ollama` |
| **P4** | `skills/` loader, example skill, `context/summarise` strategy |
| **P5** | Polish: streaming render, token counter, `tau sessions` commands |
