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

## 10. Mid-stream steer & follow-up queue

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

## 11. Providers

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

## 12. File & shell tools (used by the agent automatically)

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

## 13. Tests

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

# Quick pass/fail summary
python -m pytest tests/ -q
```

---

## 14. Gap vs pi/mono — features not yet in tau

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
| **P4** | Non-interactive pipe mode | `echo "prompt" \| tau run` — fully headless, JSON output, scriptable |
| **P4** | Structured JSON output | `tau run --json "..."` — emit agent response as machine-readable JSON |
| **P4** | `--max-cost` budget guard | Hard-stop if a session exceeds a cost ceiling |
| **P4** | Extension hot-reload | Reload extensions without restarting tau (`/reload-extensions`) |
| **P5** | TUI / split-pane view | Side-by-side file diff + chat panel (like Cursor's composer) |
| **P5** | Multi-agent orchestration | Spawn sub-agents for parallelisable sub-tasks |
| **P5** | RAG / codebase index | Embed the codebase, retrieve relevant chunks as context automatically |

**Priority legend:** P1 = quick win, P5 = large effort / scope creep.
