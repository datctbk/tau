# tau — Run Guide
> Every command you need to exercise every feature tau has today.

---

## 0. Setup

```bash
# Install in editable mode (once)
cd /Users/trantandat/Documents/tau
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your API key (OpenAI default)
export OPENAI_API_KEY="sk-..."

# Initialise ~/.tau dirs + default config
tau run --help          # triggers ensure_tau_home()
```

---

## 1. Single-shot mode

```bash
# Basic prompt
tau run "what is 2+2?"

# Choose provider / model inline
tau run -p openai -m gpt-4o-mini "summarise this file" 
tau run -p google -m gemini-2.0-flash "explain recursion"
tau run -p ollama -m llama3 "write a hello world in Go"

# Change workspace root (all file tools are sandboxed to it)
tau run -w ~/myproject "list all python files"

# Disable shell confirmation prompts
tau run --no-confirm "run pytest and show me the failures"

# Verbose mode — shows thinking tokens + debug logs
tau run -v "refactor foo.py"

# Output modes (see section 10 for details)
tau run --mode print "what is 2+2?"        # plain text, no spinners
tau run --mode json  "list files"           # JSONL event stream
tau run -P "summarise foo.py"               # shorthand for --mode print

# Prompt templates (see section 12 for details)
tau run -T code-review --var file=foo.py
tau run -T explain --var topic=recursion --var lang=python
```

---

## 2. Interactive REPL

```bash
# Start REPL (default provider/model from config)
tau run

# Start REPL with specific provider
tau run -p ollama -m llama3

# Resume a previous session in REPL
tau run -s <session-id-or-prefix>

# Name a new session
tau run --session-name "auth refactor"
```

REPL input features (see section 13 for details):
- **`@file`** — inline file contents into the prompt (`review @src/main.py`)
- **Tab** — complete slash commands and `@file` paths
- **`Ctrl-V`** — paste clipboard image (macOS)
- **`!command`** — run shell commands directly (`!git status`)

---

## 3. REPL slash commands

Once inside the REPL (`tau run`):

```
/help                        show all commands (built-in + extension)

# Follow-up queue — prompts processed one-by-one after each turn
/queue write the tests       add "write the tests" to the queue
/queue                       show how many items are queued

# Mid-stream steer — interrupt the current streaming response
/steer focus only on auth    send steer while agent is streaming
/clear                       cancel a pending steer

# Prompt templates (see section 12)
/prompt code-review file=foo.py    expand template and send as prompt
/prompts                           list available templates

# Built-in extension commands
/wc hello world              → words=2, lines=1, chars=11
/json {"a":1,"b":2}          → pretty-printed JSON

exit                         quit (also Ctrl-D)
```

---

## 4. Config

```bash
# Show current effective config
tau config show

# Set values (written to ~/.tau/config.toml [defaults])
tau config set provider openai
tau config set model gpt-4o-mini
tau config set max_tokens 16384
tau config set max_turns 30
tau config set trim_strategy sliding_window   # or: summarise
```

`~/.tau/config.toml` full example:

```toml
[defaults]
provider      = "openai"
model         = "gpt-4o"
max_tokens    = 8192
max_turns     = 20
trim_strategy = "sliding_window"

[providers.openai]
api_key  = "sk-..."
base_url = "https://api.openai.com/v1"

[providers.google]
api_key = "..."

[providers.ollama]
base_url = "http://localhost:11434"

[tools.shell]
require_confirmation = true
timeout = 30
allowed_commands = []        # empty = all allowed

[extensions]
paths    = ["~/.tau/extensions"]
disabled = ["word_count"]    # opt-out a built-in
```

Environment variable overrides (no config file needed):

```bash
export TAU_PROVIDER=openai
export TAU_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export OLLAMA_BASE_URL=http://localhost:11434
export TAU_SHELL_REQUIRE_CONFIRMATION=false
export TAU_SHELL_TIMEOUT=60
```

---

## 5. Sessions

```bash
# List all saved sessions
tau sessions list

# Show message history for a session
tau sessions show <id>
tau sessions show abc123      # prefix is fine

# Delete a session
tau sessions delete <id>
tau sessions delete <id> -y   # skip confirmation

# Resume a session (in REPL)
tau run -s <id>
```

---

## 6. Session branching (fork)

```bash
# See which user messages you can fork from
tau sessions fork-points <id>

# Fork session at message index 4 (creates a new session, stops there)
tau sessions fork <id> 4

# Fork and immediately resume in REPL
tau sessions fork <id> 4 --resume

# Fork with a custom name
tau sessions fork <id> 4 --name "try approach B"

# List direct children (branches) of a session
tau sessions branches <id>
```

---

## 7. Extensions

```bash
# List all loaded extensions (built-ins + user)
tau extensions list

# Inspect one extension in detail
tau extensions show word_count
tau extensions show pretty_json
```

**Installing a custom extension:**

```bash
# Drop a single-file extension into ~/.tau/extensions/
cp my_ext.py ~/.tau/extensions/

# Or as a package
mkdir -p ~/.tau/extensions/my_pkg
cp extension.py ~/.tau/extensions/my_pkg/

# It loads automatically next time tau starts — verify:
tau extensions list
```

**Disabling a built-in extension** (in `~/.tau/config.toml`):

```toml
[extensions]
disabled = ["word_count", "pretty_json"]
```

---

## 8. Auto-compaction

Compaction triggers automatically when the context fills up. To see it in action:

```bash
# Set a tiny max_tokens to force compaction quickly
tau run -v --no-confirm
# Then just keep chatting — you'll see:
#   ⟳ compacting context  (N tokens)
#   ✓ context compacted   N → M tokens  (saved K)
```

Force it via config:

```toml
# ~/.tau/config.toml
[defaults]
max_tokens = 2000           # small window → compaction triggers fast
```

---

## 9. Auto-retry

Retry fires automatically on rate-limit / 5xx / network errors. With verbose mode you see:

```
↻ retrying (attempt 2/3)  in 2.0s  — rate limit exceeded
```

No command needed — it's always on. To disable:

```bash
# Not exposed via CLI yet — set in AgentConfig directly or via extension
```

---

## 10. Output modes

tau supports four output modes: **interactive** (default), **print**, **json**, and **rpc**.

### Interactive (default)

Rich TUI with spinners, live Markdown rendering, and coloured tool output. Used when you run `tau run` in a terminal.

### Print mode

Plain text output with no spinners or formatting. Ideal for piping into other tools.

```bash
# Explicit flag
tau run --mode print "what is 2+2?"
tau run -P "summarise foo.py"

# Piped stdin — automatically uses print mode
echo "explain recursion" | tau run
cat prompt.txt | tau run -p google -m gemini-2.5-pro
```

### JSON mode (JSONL)

Emits one JSON object per line for every event the agent produces. Useful for scripting, logging, and building tooling on top of tau.

```bash
tau run --mode json "list files in src/"
```

Example output:

```jsonl
{"type":"text","content":"Here are the files…"}
{"type":"tool_call","tool":"list_files","args":{"path":"src/"}}
{"type":"tool_result","tool":"list_files","result":"foo.py\nbar.py"}
{"type":"text","content":"The directory contains foo.py and bar.py."}
{"type":"finished","usage":{"input_tokens":320,"output_tokens":85}}
```

### RPC mode (JSONL over stdio)

Starts a long-running process that reads JSON requests from stdin and writes JSON responses to stdout. Designed for embedding tau in editors, bots, or other processes.

```bash
tau run --mode rpc
tau run --mode rpc -p google -m gemini-2.5-pro
```

**Protocol:** LF-delimited JSONL. The server emits `{"type": "ready"}` when idle.

**Client → Server requests:**

| type | fields | description |
|------|--------|-------------|
| `prompt` | `text`, `images` (optional) | Send a message; server streams back events then `{"type": "ready"}` |
| `steer` | `text` | Inject a mid-stream steering message |
| `enqueue` | `text` | Queue a follow-up message |
| `session_info` | — | Get current session metadata |
| `exit` | — | Shut down the RPC server |

**Example exchange:**

```jsonl
→ {"type": "prompt", "text": "list files in src/"}
← {"type": "text_delta", "text": "Here are"}
← {"type": "text_delta", "text": " the files:"}
← {"type": "tool_call", "call": {"id": "...", "name": "list_files", "arguments": {"path": "src/"}}}
← {"type": "tool_result", "result": {"tool_call_id": "...", "content": "foo.py\nbar.py", "is_error": false}}
← {"type": "turn_complete", "usage": {"input_tokens": 320, "output_tokens": 85, ...}}
← {"type": "ready"}
→ {"type": "session_info"}
← {"type": "session_info", "id": "...", "model": "gpt-4o", "messages": 3, ...}
→ {"type": "exit"}
← {"type": "exit", "status": "ok"}
```

---

## 11. Programmatic SDK

Embed tau in your own Python application with the SDK. No CLI, no TUI — just import and call.

### Basic usage

```python
from tau import create_session

with create_session(provider="openai", model="gpt-4o") as session:
    for event in session.prompt("What files are in the current directory?"):
        print(event.to_dict())
```

### Non-streaming

```python
events = session.prompt_sync("Summarize README.md")
```

### Options

```python
session = create_session(
    provider="google",
    model="gemini-2.5-pro",
    system_prompt="You are a helpful assistant.",
    workspace="/path/to/project",
    in_memory=True,           # no disk I/O for sessions
    load_skills=False,         # skip skill discovery
    load_extensions=False,     # skip extension discovery
    load_context_files=False,  # skip AGENTS.md / SYSTEM.md
    shell_confirm=False,       # no shell confirmation prompts
)
```

### In-memory sessions

By default `create_session(in_memory=True)` uses `InMemorySessionManager` — everything stays in memory. Pass `in_memory=False` or your own `session_manager=` to persist to disk.

```python
from tau import InMemorySessionManager, create_session

sm = InMemorySessionManager()
session = create_session(session_manager=sm, ...)
```

### Steering & follow-ups

```python
session.steer("Actually, focus on security issues")
session.enqueue("Now review the tests too")
```

### RPC from code

```python
from tau import create_session, run_rpc

session = create_session(provider="openai", model="gpt-4o")
run_rpc(session)  # reads stdin, writes stdout
```

---

## 12. System prompt override (`.tau/SYSTEM.md`)

Place a `SYSTEM.md` file in your project's `.tau/` directory to **replace** the default system prompt entirely. This lets you tailor the agent's persona or rules per-project.

```bash
mkdir -p .tau
cat > .tau/SYSTEM.md << 'EOF'
You are a security-focused code reviewer.
Always check for OWASP Top 10 vulnerabilities.
Never suggest code that uses eval() or exec().
EOF

tau run "review auth.py"
```

- When `.tau/SYSTEM.md` exists it **replaces** the built-in system prompt.
- `AGENTS.md` / `CLAUDE.md` context files still append after the system prompt as usual.
- Delete the file to go back to the default behaviour.

---

## 13. Prompt templates

Reusable Markdown prompts with `{{variable}}` placeholders. Store them in `.tau/prompts/` (project-level) or `~/.tau/prompts/` (global). Project templates take priority over global ones with the same name.

### Creating a template

```bash
mkdir -p .tau/prompts

cat > .tau/prompts/code-review.md << 'EOF'
Review the file **{{file}}** for:
- bugs and logic errors
- security issues
- performance concerns
Suggest concrete fixes.
EOF

cat > .tau/prompts/explain.md << 'EOF'
Explain {{topic}} in {{lang}} with examples.
Keep it beginner-friendly.
EOF
```

### Using templates from the CLI

```bash
# Expand a template and run it
tau run -T code-review --var file=foo.py
tau run -T explain --var topic=recursion --var lang=python

# Combine with other flags
tau run -T code-review --var file=auth.py -p google -m gemini-2.5-pro --mode print
```

### Using templates in the REPL

```
/prompts                               list all available templates
/prompt code-review file=foo.py        expand and send as prompt
/prompt explain topic=async lang=go    expand and send
```

### Managing templates

```bash
# List all discovered templates (project + global)
tau prompts list

# Show the raw Markdown and variables for a template
tau prompts show code-review
```

Missing variables are left as `{{name}}` in the expanded text so the LLM can still see them.

---

## 14. Editor richness

### `@file` references

Mention a file path with `@` to inline its contents into the prompt. Works in both the REPL and single-shot mode.

```bash
# Single-shot — file contents are expanded before sending to the LLM
tau run "review @src/auth.py for security issues"
tau run "compare @foo.py and @bar.py"

# REPL — same syntax
you review @utils.py
  📎 1 file inlined: utils.py
```

- Paths are relative to the workspace root (or absolute, if inside the workspace).
- Files outside the workspace are **blocked** (left as-is).
- Files larger than 256 KB are skipped with a note.

### Tab completion

Press **Tab** in the REPL to complete:

- **Slash commands**: `/he⇥` → `/help`
- **`@file` paths**: `@src/m⇥` → `@src/main.py`

Directories are shown with a trailing `/`. Hidden files (`.dotfiles`) are only shown when the prefix starts with `.`.

### Image paste (`Ctrl-V`)

On macOS, press **Ctrl-V** in the REPL to paste an image from the clipboard. The image is saved to a temp file and staged automatically:

```
you describe this screenshot
  📷 image pasted from clipboard → tmpXXXX.png
```

You can also stage images manually with `/image <path>` or `--image <path>`.

### Inline shell (`!command`)

Run shell commands directly from the REPL without going through the agent:

```
!ls src/
  main.py  utils.py  auth.py

!git status
  On branch main
  nothing to commit, working tree clean

!python -m pytest tests/ -q
  450 passed in 7.5s
```

- Commands run in the workspace root directory.
- Output is displayed directly in the REPL output pane.
- These bypass the agent entirely — useful for quick checks.

---

## 15. Mid-stream steer & follow-up queue

```bash
tau run          # start REPL

# --- in REPL ---
# Type a prompt and hit enter, then while it's streaming type:
/steer actually make it async instead

# Queue several follow-ups before the first turn even starts:
/queue now write the tests
/queue now run the tests with pytest
# Then type the first prompt:
refactor auth.py
# tau will run all three turns automatically
```

---

## 16. Providers

```bash
# OpenAI (default)
tau run -p openai -m gpt-4o "hello"
tau run -p openai -m gpt-4o-mini "hello"
tau run -p openai -m o3-mini "hello"

# Google Gemini
export GOOGLE_API_KEY=...
tau run -p google -m gemini-2.0-flash "hello"
tau run -p google -m gemini-2.5-pro "hello"

# Ollama (local, no key)
# First: ollama pull llama3
tau run -p ollama -m llama3 "hello"
tau run -p ollama -m codellama "explain this code"
tau run -p ollama -m mistral "hello"
```

---

## 17. File & shell tools (used by the agent automatically)

The agent calls these itself during a task. You can also ask for them explicitly:

```bash
tau run "read pyproject.toml and tell me the dependencies"
tau run "list all .py files under tau/core/"
tau run "search for all TODO comments in the codebase"
tau run "write a file called hello.txt with the content 'hi'"
tau run "edit tau/cli.py: replace 'gpt-4o' with 'gpt-4o-mini'"
tau run --no-confirm "run: pytest tests/ -q"
```

---

## 18. Tests

```bash
# Full suite
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_agent.py -v
python -m pytest tests/test_compaction.py -v
python -m pytest tests/test_context.py -v
python -m pytest tests/test_extensions.py -v
python -m pytest tests/test_fs_tools.py -v
python -m pytest tests/test_retry.py -v
python -m pytest tests/test_session.py -v
python -m pytest tests/test_steering.py -v
python -m pytest tests/test_tool_registry.py -v

# New feature tests
python -m pytest tests/test_output_modes.py -v
python -m pytest tests/test_prompts.py -v
python -m pytest tests/test_editor.py -v
python -m pytest tests/test_sdk_rpc.py -v

# Quick pass/fail summary
python -m pytest tests/ -q
```

---

## 19. Gap vs pi/mono — features not yet in tau

The following features exist in full-featured coding agents (pi, Cursor, Claude Code, Aider) but are **not yet implemented** in tau:

| # | Feature | What it means |
|---|---------|---------------|
| **P1** | `/compact` slash command | Manually trigger compaction from the REPL instead of waiting for auto-threshold |
| **P1** | `/model <name>` slash command | Hot-swap the model mid-conversation without restarting |
| **P1** | `/clear` context reset | Wipe message history and start fresh in the same session |
| **P2** | Multi-file diff / patch tool | Apply a unified diff across multiple files atomically; show a rich before/after view |
| **P2** | `glob` + `find` tools | Recursive glob matching and `find`-style file search (richer than current `search_files`) |
| **P2** | URL / web fetch tool | `fetch_url(url)` — let the agent read documentation, issues, or web content |
| **P2** | Image / vision input | Pass screenshots or diagrams to vision-capable models (GPT-4o, Gemini) |
| **P3** | `tau init` project setup | Scaffold a per-project `.tau/` with a `CLAUDE.md`-style context file auto-loaded as system prompt |
| **P3** | Project context file | `CONTEXT.md` / `.tau/context.md` auto-injected into system prompt at startup |
| **P3** | `/cost` command | Show accumulated token cost for the current session (input+output × price per token) |
| **P3** | `/tokens` command | Show live token usage breakdown (used / budget / remaining) |
| **P3** | Thinking-level control | Per-request `thinking_budget` for Claude o1/o3 and Gemini thinking models |
| **P3** | Model cycling (`/next`) | Round-robin through a configured list of models for comparison |
| **P3** | `tau sessions export` | Export a session as Markdown or JSON for sharing |
| **P3** | `tau sessions import` | Import an exported session |
| **P3** | `tau sessions rename` | Rename an existing session |
| **P4** | Git-aware tools | `git_status`, `git_diff`, `git_log`, `git_commit` tools exposed to the agent |
| **P4** | LSP / diagnostic tool | Query a running language server for errors, hover info, go-to-definition |
| **P4** | `tau doctor` | Check API keys, provider reachability, config validity |
| ~~**P4**~~ | ~~Non-interactive pipe mode~~ | ✅ Implemented — see section 10. `echo "prompt" \| tau run` with auto print mode |
| ~~**P4**~~ | ~~Structured JSON output~~ | ✅ Implemented — see section 10. `tau run --mode json "..."` |
| **P4** | `--max-cost` budget guard | Hard-stop if a session exceeds a cost ceiling |
| **P4** | Extension hot-reload | Reload extensions without restarting tau (`/reload-extensions`) |
| **P5** | TUI / split-pane view | Side-by-side file diff + chat panel (like Cursor's composer) |
| **P5** | Multi-agent orchestration | Spawn sub-agents for parallelisable sub-tasks |
| **P5** | RAG / codebase index | Embed the codebase, retrieve relevant chunks as context automatically |

**Priority legend:** P1 = quick win, P5 = large effort / scope creep.
