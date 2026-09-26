# tau

A minimal, extensible CLI coding agent with multi-provider LLM support and tool use.

## Quick start

```bash
pip install -e .
tau "fix the bug in foo.py"
tau                        # interactive REPL
```

## Features

- **Multi-provider**: OpenAI, Anthropic Claude, Google Gemini, Ollama (local), MLX (Apple Silicon local) — swap with `-p` / `-m`
- **Built-in task primitives**: `task_create`, `task_update`, `task_list` plus plan mode (`/plan on|off|status`, `/tasks`)
- **Code index (Merkle)**: changed-file tracking, `/code-index-status`, `/code-index-refresh`, changed-only file search path
- **MCP resources/tools (minimal built-in)**: `mcp_list_resources`, `mcp_read_resource`, `mcp_list_tools`, `mcp_call_tool`, `/mcp-resources`, `/mcp-tools`
- **Output modes**: interactive (rich TUI), print (`-P`), JSON (`--mode json`), and RPC (`--mode rpc`) for process integration
- **Piped stdin**: `echo "prompt" | tau run` auto-detects non-TTY and uses print mode
- **SDK**: `from tau import create_session` — embed tau in your Python apps
- **RPC mode**: JSONL-over-stdio protocol for non-Python integrations (editors, bots, etc.)
- **System prompt override**: drop a `.tau/SYSTEM.md` in your project to replace the default prompt
- **Prompt templates**: reusable Markdown with `{{variable}}` placeholders in `.tau/prompts/`
- **Editor richness**: `@file` inlining, tab completion, `Ctrl-V` image paste, `!shell` escape
- **Sessions**: save, resume, fork, branch, and **export** conversation history (JSON or Markdown)
- **Extensions**: drop-in Python extensions with custom `/slash` commands
- **Themes**: customisable colours via `[theme]` in `~/.tau/config.toml`
- **Configurable tool set**: disable or whitelist tools via `[tools]` config
- **Auto-compaction**: automatic context trimming when the window fills up
- **Modern compaction (opt-in)**: pass `-zip` to enable pre-prune + iterative summary + quality checks; default remains legacy compaction
- **Auto-retry**: transparent retry on rate-limit and transient errors
- **File & shell tools**: sandboxed read/write/search/edit/shell with confirmation

## Providers

| Provider | Env var |
|----------|---------|
| openai   | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| google   | `GOOGLE_API_KEY` |
| ollama   | *(local, no key needed)* |
| mlx      | *(local Apple Silicon, no key needed)* |

## Environment Variables

### Provider envs

```bash
# OpenAI
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Anthropic
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Google
export GOOGLE_API_KEY="..."

# Ollama
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_TIMEOUT_SECONDS="120"

# Unsloth
export UNSLOTH_BASE_URL="http://localhost:8001/v1"
export UNSLOTH_TIMEOUT_SECONDS="120"
export UNSLOTH_STREAM_READ_TIMEOUT_SECONDS="0"
export UNSLOTH_STREAM_YIELD_EVERY_CHUNKS="0"
export UNSLOTH_STREAM_YIELD_MS="0"

# MLX
export MLX_DEVICE="gpu"
export MLX_TEMPERATURE="0.7"
export MLX_TOP_P="0.9"
export MLX_REPETITION_PENALTY="1.05"
export MLX_MEMORY_LIMIT_GB="16"
export MLX_WIRED_LIMIT_GB="12"
export MLX_CACHE_LIMIT_MB="64"
export MLX_PREFILL_STEP_SIZE="128"
export MLX_MAX_KV_SIZE="1024"
export MLX_KV_BITS="4"
export MLX_QUANTIZED_KV_START="0"
export MLX_GPU_YIELD_EVERY="1"
export MLX_GPU_YIELD_MS="8"
```

### Core `TAU_*` envs

```bash
# Core model/runtime
export TAU_PROVIDER="openai"
export TAU_MODEL="gpt-4o"
export TAU_MAX_TOKENS="110592"
export TAU_MAX_TURNS="20"
export TAU_TRIM_STRATEGY="sliding_window"
export TAU_COMPACTION_ENABLED="true"
export TAU_COMPACTION_THRESHOLD="0.60"
export TAU_SYSTEM_PROMPT="You are tau ..."
export TAU_PARALLEL_TOOLS="true"
export TAU_PARALLEL_TOOLS_MAX_WORKERS="8"
export TAU_MAX_COST="0"
export TAU_POLICY_ENABLED="true"
export TAU_POLICY_PROFILE="balanced"   # strict|balanced|dev
export TAU_PROMPT_BUDGET_ENABLED="false"
export TAU_PROMPT_BUDGET_MAX_INPUT_TOKENS="3200"
export TAU_PROMPT_BUDGET_OUTPUT_RESERVE="1000"
export TAU_PROMPT_BUDGET_MAX_TOOLS_TOTAL="12"
export TAU_DYNAMIC_PROMPT_BUILDER_ENABLED="false"
export TAU_CREDENTIAL_POOL_ENABLED="false"
export TAU_MINIMAL_MODE="false"

# Shell tools
export TAU_SHELL_REQUIRE_CONFIRMATION="true"
export TAU_SHELL_TIMEOUT="30"
export TAU_SHELL_ALLOWED_COMMANDS=""
export TAU_SHELL_USE_PERSISTENT_SHELL="false"

# Skills/extensions
export TAU_SKILLS_PATHS=""
export TAU_SKILLS_DISABLED=""
export TAU_EXTENSIONS_PATHS=""
export TAU_EXTENSIONS_DISABLED=""

# Thinking budgets
export TAU_THINKING_BUDGETS_MINIMAL="1024"
export TAU_THINKING_BUDGETS_LOW="2048"
export TAU_THINKING_BUDGETS_MEDIUM="8192"
export TAU_THINKING_BUDGETS_HIGH="16384"
export TAU_THINKING_BUDGETS_XHIGH="32768"

# Smart routing
export TAU_SMART_ROUTING_ENABLED="false"
export TAU_SMART_ROUTING_CHEAP_MODEL_PROVIDER="unsloth"
export TAU_SMART_ROUTING_CHEAP_MODEL="gemma-4-26B-A4B-it-GGUF"
export TAU_SMART_ROUTING_MAX_SIMPLE_CHARS="160"
export TAU_SMART_ROUTING_MAX_SIMPLE_WORDS="28"

# Capabilities
export TAU_CAPABILITIES_PROMPT_CACHING="true"
export TAU_CAPABILITIES_RATE_LIMIT_TRACKING="true"
export TAU_CAPABILITIES_SMART_ROUTING="true"
export TAU_CAPABILITIES_USAGE_PRICING="true"
export TAU_CAPABILITIES_CREDENTIAL_POOL="true"

# Theme/tools
export TAU_THEME_PRESET=""
export TAU_THEME_USER_COLOR="cyan"
export TAU_THEME_ASSISTANT_COLOR="green"
export TAU_THEME_TOOL_COLOR="yellow"
export TAU_THEME_SYSTEM_COLOR="dim"
export TAU_THEME_ERROR_COLOR="red"
export TAU_THEME_ACCENT_COLOR="cyan"
export TAU_THEME_SUCCESS_COLOR="green"
export TAU_THEME_WARNING_COLOR="yellow"
export TAU_THEME_BORDER_STYLE="dim"
export TAU_TOOLS_DISABLED=""
export TAU_TOOLS_ENABLED_ONLY=""
```

### Semantic retrieval / indexing envs (opt-in)

Tau keeps the default retrieval path **lexical/minimal**.

- `TAU_SEMANTIC_RETRIEVAL=1`: enables semantic retrieval mode (master switch)
- `TAU_EMBEDDING_CACHE_ENABLED=1`: enables embedding-cache participation
- `TAU_EMBEDDING_CACHE_MODEL=<name>`: embedding cache model key (default: `default`)
- `TAU_EMBEDDING_CACHE_DB_PATH=<path>`: custom embedding cache DB path
- `TAU_SEMANTIC_STORE_ENABLED=1`: ingest changed chunks into local semantic store (default on when semantic mode is on)
- `TAU_SEMANTIC_STORE_DB_PATH=<path>`: custom semantic store DB path (default: `~/.tau/semantic_store.db`)
- `TAU_SEMANTIC_MODEL=<name>`: semantic vector model key (default: `local-hash-v1`)
- `TAU_SEMANTIC_LEXICAL_WEIGHT=<float>`: hybrid lexical weight (default `0.45`)
- `TAU_SEMANTIC_VECTOR_WEIGHT=<float>`: hybrid vector weight (default `0.55`)
- `TAU_SEMANTIC_EMBEDDINGS_ENABLED=1`: enables embedder component (requires master switch)
- `TAU_SEMANTIC_VECTOR_INDEX_ENABLED=1`: enables vector-index component (requires master switch)

Default behavior (no flags): lexical path only, semantic components off.

### Voice / gateway / UI / tracing envs

```bash
# Voice / STT
export TAU_STT_COMMAND=""
export TAU_GATEWAY_STT_COMMAND=""
export TAU_STT_RECORD_COMMAND=""
export TAU_STT_RECORD_SECONDS="8"
export TAU_STT_RECORD_HOLD_COMMAND=""

# Gateway entrypoint override
export TAU_GATEWAY_ENTRYPOINT=""
export TAU_GATEWAY_ALLOW_CWD_ENTRYPOINT=""

# TUI output limits
export TAU_TUI_MAX_LINES="8000"
export TAU_TUI_MAX_CHARS="1200000"

# Trace controls
export TAU_TRACE_FULL="false"
export TAU_TRACE_MAX_MESSAGE_CHARS="0"
export TAU_TRACE_MAX_TOOL_ARGS_CHARS="0"
export TAU_TRACE_MAX_RESPONSE_CHARS="0"
export TAU_TRACE_MAX_THINKING_CHARS="5000"

# Optional home override for credential pool path
export TAU_HOME="$HOME/.tau"
```

See [RUN.md](RUN.md) for the full usage guide and [DESIGN.md](DESIGN.md) for the architecture.
