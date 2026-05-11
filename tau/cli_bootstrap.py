"""CLI bootstrap helpers (non-UI path)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from tau.config import TauConfig
from tau.core.agent import Agent
from tau.core.context import ContextManager, configure_context
from tau.core.extension import ExtensionRegistry
from tau.core.prompt_layers import PromptLayer, apply_prompt_layers
from tau.core.session import SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig
from tau.providers import get_provider
from tau.skills import SkillLoader
from tau.tools import register_builtin_tools
from tau.tools.fs import configure_fs
from tau.tools.shell import configure_shell

logger = logging.getLogger(__name__)


def apply_minimal_profile(tau_config: TauConfig) -> TauConfig:
    """Apply strict minimal defaults in-place."""
    tau_config.minimal_mode = True
    tau_config.dynamic_prompt_builder_enabled = False
    tau_config.prompt_budget_enabled = False
    tau_config.smart_routing.enabled = False
    tau_config.credential_pool_enabled = False
    tau_config.capabilities.prompt_caching = False
    tau_config.capabilities.rate_limit_tracking = False
    tau_config.capabilities.smart_routing = False
    tau_config.capabilities.usage_pricing = False
    tau_config.capabilities.credential_pool = False
    return tau_config


def make_agent_config(
    tau_config: TauConfig,
    provider: str | None,
    model: str | None,
    think: str | None,
    no_confirm: bool,
    workspace: str,
    no_parallel: bool = False,
    persistent_shell: bool = False,
    max_cost: float | None = None,
    topk: int = 0,
    code_index: bool = False,
    dynamic_prompt_builder: bool | None = None,
    prompt_budget: bool | None = None,
    minimal: bool = False,
) -> AgentConfig:
    if provider:
        tau_config.provider = provider
    if no_confirm:
        tau_config.shell.require_confirmation = False
    if persistent_shell:
        tau_config.shell.use_persistent_shell = True
    if prompt_budget is not None:
        tau_config.prompt_budget_enabled = bool(prompt_budget)
    if dynamic_prompt_builder is not None:
        tau_config.dynamic_prompt_builder_enabled = bool(dynamic_prompt_builder)
    if minimal:
        apply_minimal_profile(tau_config)
    return AgentConfig(
        provider=tau_config.provider,
        model=model or tau_config.model,
        thinking_level=think or "off",
        thinking_budgets=tau_config.thinking_budgets.model_dump(),
        max_tokens=tau_config.max_tokens,
        max_turns=tau_config.max_turns,
        system_prompt=tau_config.system_prompt,
        trim_strategy=tau_config.trim_strategy,
        compaction_enabled=tau_config.compaction_enabled,
        compaction_threshold=tau_config.compaction_threshold,
        workspace_root=str(Path(workspace).resolve()),
        parallel_tools=tau_config.parallel_tools and not no_parallel,
        parallel_tools_max_workers=tau_config.parallel_tools_max_workers,
        max_cost=max_cost if max_cost is not None else tau_config.max_cost,
        memory_topk=max(0, topk),
        code_index_enabled=bool(code_index),
        policy_enabled=tau_config.policy_enabled,
        policy_profile=tau_config.policy_profile,
        prompt_budget_enabled=bool(tau_config.prompt_budget_enabled),
        prompt_budget_max_input_tokens=max(512, int(tau_config.prompt_budget_max_input_tokens)),
        prompt_budget_output_reserve=max(0, int(tau_config.prompt_budget_output_reserve)),
        prompt_budget_max_tools_total=max(1, int(tau_config.prompt_budget_max_tools_total)),
        dynamic_prompt_builder_enabled=bool(tau_config.dynamic_prompt_builder_enabled),
        smart_routing_config=(
            tau_config.smart_routing
            if (tau_config.smart_routing.enabled and tau_config.capabilities.smart_routing)
            else None
        ),
        capabilities=tau_config.capabilities.model_dump(),
        minimal_mode=bool(tau_config.minimal_mode),
    )


def _load_prompt_layers(
    *,
    context: ContextManager,
    workspace_root: str,
    ext_registry: ExtensionRegistry,
    minimal_mode: bool,
    code_index_enabled: bool,
) -> None:
    layers: list[PromptLayer] = []
    if not minimal_mode:
        from tau.context_files import load_context_files

        context_text = load_context_files(workspace_root)
        if context_text:
            layers.append(PromptLayer(name="workspace:context-files", content=context_text, priority=55))
        if code_index_enabled:
            try:
                from tau.core.code_index import (
                    refresh_code_index,
                )
                stats = refresh_code_index(workspace_root)
                changes = stats["changes"]

                changed = changes.changed
                deleted = changes.deleted
                preview_n = 30
                lines = [
                    "## Code Index Delta (Merkle)",
                    f"- added: {len(changes.added)}",
                    f"- modified: {len(changes.modified)}",
                    f"- deleted: {len(deleted)}",
                    f"- unchanged: {changes.unchanged_count}",
                    f"- files indexed: {int(stats.get('file_count', 0))}",
                    f"- scan duration: {int(stats.get('duration_ms', 0))}ms",
                ]
                if changed:
                    lines.append("- changed paths:")
                    for p in changed[:preview_n]:
                        lines.append(f"  - {p}")
                    if len(changed) > preview_n:
                        lines.append(f"  - ... and {len(changed) - preview_n} more")
                if deleted:
                    lines.append("- deleted paths:")
                    for p in deleted[: min(10, len(deleted))]:
                        lines.append(f"  - {p}")
                lines.append(
                    "Use this delta to prioritize analysis on changed files before broad repo scans."
                )
                layers.append(
                    PromptLayer(
                        name="workspace:code-index-delta",
                        content="\n".join(lines),
                        priority=57,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Code index delta generation failed: %s", exc)
    layers.extend(ext_registry.prompt_layers())
    apply_prompt_layers(context, layers)


def build_agent(
    *,
    tau_config: TauConfig,
    agent_config: AgentConfig,
    session_manager: SessionManager,
    session_name: str | None,
    resume_id: str | None,
    confirm_hook: Callable[[str], bool] | None = None,
    policy_approval_hook: Callable[[str], bool] | None = None,
    steering: SteeringChannel | None = None,
    tools_filter: str | None = None,
    print_fn: Callable[[str], None] | None = None,
    provider_factory: Callable[[TauConfig, AgentConfig], object] | None = None,
) -> tuple[Agent, ExtensionRegistry]:
    registry = ToolRegistry()
    register_builtin_tools(registry)

    if tools_filter is not None:
        requested = {t.strip() for t in tools_filter.split(",") if t.strip()}
        for name in list(registry.names()):
            if name not in requested:
                registry.unregister(name)
    elif tau_config.tools.enabled_only:
        allowed = set(tau_config.tools.enabled_only)
        for name in list(registry.names()):
            if name not in allowed:
                registry.unregister(name)
    elif tau_config.tools.disabled:
        for name in tau_config.tools.disabled:
            registry.unregister(name)

    configure_shell(
        require_confirmation=tau_config.shell.require_confirmation,
        timeout=tau_config.shell.timeout,
        allowed_commands=tau_config.shell.allowed_commands,
        use_persistent_shell=tau_config.shell.use_persistent_shell,
        confirm_hook=confirm_hook,
        workspace_root=agent_config.workspace_root,
    )
    configure_fs(workspace_root=agent_config.workspace_root)
    configure_context(
        ollama_base_url=tau_config.ollama.base_url,
        ollama_model=agent_config.model,
    )

    context = ContextManager(agent_config)
    from tau.context_files import load_system_prompt_override

    system_override = load_system_prompt_override(agent_config.workspace_root)
    if system_override is not None:
        agent_config.system_prompt = system_override
        for m in context._messages:
            if m.role == "system":
                m.content = system_override
                break

    ext_registry = ExtensionRegistry(
        extra_paths=[] if agent_config.minimal_mode else list(tau_config.extensions.paths),
        disabled=tau_config.extensions.disabled,
        include_builtins=not agent_config.minimal_mode,
    )

    if not agent_config.minimal_mode:
        try:
            from tau.packages import PackageManager as _PM

            _pm = _PM()
            _pkg_skill_paths = _pm.get_resource_paths("skills")
            _pkg_extension_paths = _pm.get_resource_paths("extensions")
            ext_registry = ExtensionRegistry(
                extra_paths=list(tau_config.extensions.paths) + _pkg_extension_paths,
                disabled=tau_config.extensions.disabled,
                include_builtins=True,
            )
        except Exception:
            _pkg_skill_paths = []
        loader = SkillLoader(
            extra_paths=list(tau_config.skills.paths) + _pkg_skill_paths,
            disabled=tau_config.skills.disabled,
        )
        loader.load_into(registry, context)

    loaded_exts = ext_registry.load_all(
        registry=registry,
        context=context,
        steering=steering,
        console_print=print_fn or (lambda _s: None),
        agent_config=agent_config,
    )
    _load_prompt_layers(
        context=context,
        workspace_root=agent_config.workspace_root,
        ext_registry=ext_registry,
        minimal_mode=agent_config.minimal_mode,
        code_index_enabled=bool(agent_config.code_index_enabled),
    )
    if print_fn:
        if loaded_exts:
            print_fn("[dim]Loaded extensions: " + ", ".join(loaded_exts) + "[/dim]")
        else:
            print_fn("[dim]Loaded extensions: (none)[/dim]")

    if resume_id:
        session = session_manager.load(resume_id)
        context.restore(session.messages)
        if print_fn:
            print_fn(
                f"[dim]Resumed session [bold]{session.id[:8]}[/bold]"
                + (f" ({session.name})" if session.name else "")
                + "[/dim]"
            )
    else:
        session = session_manager.new_session(agent_config, name=session_name)

    if tau_config.credential_pool_enabled and tau_config.capabilities.credential_pool:
        from tau.config import TAU_HOME
        from tau.core.credential_pool import CredentialPool

        pool_path = Path(TAU_HOME) / "credentials" / "pool.json"
        if pool_path.exists():
            try:
                pool = CredentialPool.load(pool_path)
                cred = pool.select(provider=tau_config.provider)
                if cred:
                    if tau_config.provider == "openai":
                        tau_config.openai.api_key = cred.api_key
                        if cred.base_url:
                            tau_config.openai.base_url = cred.base_url
                    elif tau_config.provider == "google":
                        tau_config.google.api_key = cred.api_key
                    elif tau_config.provider == "anthropic":
                        tau_config.anthropic.api_key = cred.api_key
                        if cred.base_url:
                            tau_config.anthropic.base_url = cred.base_url
                logger.debug("Credential pool selected key for %s", tau_config.provider)
            except Exception as e:
                logger.warning("Failed to load credential pool: %s", e)

    provider = (provider_factory or get_provider)(tau_config, agent_config)
    agent = Agent(
        config=agent_config,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=session_manager,
        steering=steering,
        cost_calculator=tau_config.calculate_cost,
        policy_approval_hook=policy_approval_hook,
    )
    return agent, ext_registry
