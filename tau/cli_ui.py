"""CLI UI/mode dispatch helpers."""

from __future__ import annotations

import sys
from typing import Callable

from tau.core.agent import Agent
from tau.core.extension import ExtensionRegistry


def run_mode(
    *,
    mode: str | None,
    prompt: str | None,
    image: tuple[str, ...],
    agent: Agent,
    ext_registry: ExtensionRegistry,
    show_thinking: bool,
    verbose: bool,
    render_json: Callable[[Agent, str, list[str] | None, ExtensionRegistry | None], None],
    render_print: Callable[[Agent, str, list[str] | None, ExtensionRegistry | None], None],
    render_stream: Callable[[Agent, str, bool, list[str] | None, ExtensionRegistry | None, bool], None],
    render_repl: Callable[..., None],
    repl_kwargs: dict,
) -> None:
    """Run one of the CLI interaction modes."""
    images = list(image) if image else None
    if mode == "json":
        if not prompt:
            print("Error: --mode json requires a prompt.", file=sys.stderr)
            sys.exit(1)
        render_json(agent, prompt, images, ext_registry)
        return
    if mode == "print":
        if not prompt:
            print("Error: --mode print requires a prompt.", file=sys.stderr)
            sys.exit(1)
        render_print(agent, prompt, images, ext_registry)
        return
    if prompt:
        render_stream(agent, prompt, verbose, images, ext_registry, show_thinking)
        return
    render_repl(
        agent,
        **repl_kwargs,
    )

