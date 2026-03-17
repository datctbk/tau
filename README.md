# tau

A minimal, extensible CLI coding agent with multi-provider LLM support and tool use.

## Quick start

```bash
pip install -e .
tau "fix the bug in foo.py"
tau                        # interactive REPL
```

## Providers

| Provider | Env var |
|----------|---------|
| openai   | `OPENAI_API_KEY` |
| google   | `GOOGLE_API_KEY` |
| ollama   | *(local, no key needed)* |

See `DESIGN.md` for the full architecture.
