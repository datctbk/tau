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

See [RUN.md](RUN.md) for the full usage guide and [DESIGN.md](DESIGN.md) for the architecture.
