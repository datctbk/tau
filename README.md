# tau

A minimal, extensible CLI coding agent with multi-provider LLM support and tool use.

## Quick start

```bash
pip install -e .
tau "fix the bug in foo.py"
tau                        # interactive REPL
```

## Features

- **Multi-provider**: OpenAI, Google Gemini, Ollama (local) — swap with `-p` / `-m`
- **Output modes**: interactive (rich TUI), print (`-P`), JSON (`--mode json`), and RPC (`--mode rpc`) for process integration
- **Piped stdin**: `echo "prompt" | tau run` auto-detects non-TTY and uses print mode
- **SDK**: `from tau import create_session` — embed tau in your Python apps
- **RPC mode**: JSONL-over-stdio protocol for non-Python integrations (editors, bots, etc.)
- **System prompt override**: drop a `.tau/SYSTEM.md` in your project to replace the default prompt
- **Prompt templates**: reusable Markdown with `{{variable}}` placeholders in `.tau/prompts/`
- **Editor richness**: `@file` inlining, tab completion, `Ctrl-V` image paste, `!shell` escape
- **Sessions**: save, resume, fork, and branch conversation history
- **Extensions**: drop-in Python extensions with custom `/slash` commands
- **Auto-compaction**: automatic context trimming when the window fills up
- **Auto-retry**: transparent retry on rate-limit and transient errors
- **File & shell tools**: sandboxed read/write/search/edit/shell with confirmation

## Providers

| Provider | Env var |
|----------|---------|
| openai   | `OPENAI_API_KEY` |
| google   | `GOOGLE_API_KEY` |
| ollama   | *(local, no key needed)* |

See [RUN.md](RUN.md) for the full usage guide and [DESIGN.md](DESIGN.md) for the architecture.
