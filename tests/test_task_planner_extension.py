from __future__ import annotations

from tau.core.context import ContextManager
from tau.core.extension import ExtensionContext
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig, ToolCall
from tau.extensions.task_planner import TaskPlannerExtension


def _cfg(tmp_path) -> AgentConfig:
    return AgentConfig(
        provider="openai",
        model="gpt-4o",
        workspace_root=str(tmp_path),
        compaction_enabled=False,
        retry_enabled=False,
    )


def _ctx(tmp_path):
    reg = ToolRegistry()
    ctx = ContextManager(_cfg(tmp_path))
    steering = SteeringChannel()
    prints: list[str] = []
    ext_ctx = ExtensionContext(registry=reg, context=ctx, steering=steering, console_print=prints.append, agent_config=_cfg(tmp_path))
    return reg, ctx, ext_ctx, prints


def test_task_planner_registers_tools_and_commands(tmp_path):
    reg, _, ext_ctx, _ = _ctx(tmp_path)
    ext = TaskPlannerExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)
    assert "task_create" in reg.names()
    assert "task_update" in reg.names()
    assert "task_list" in reg.names()

    slash = {c.name for c in ext.slash_commands()}
    assert "tasks" in slash
    assert "plan" in slash


def test_task_create_update_list_lifecycle(tmp_path):
    reg, _, ext_ctx, _ = _ctx(tmp_path)
    ext = TaskPlannerExtension()
    ext.on_load(ext_ctx)
    for tool in ext.tools():
        reg.register(tool)

    created = reg.dispatch(ToolCall(id="1", name="task_create", arguments={"title": "Implement parser"}))
    assert "Created task t_" in created.content

    listed = reg.dispatch(ToolCall(id="2", name="task_list", arguments={}))
    assert "Implement parser" in listed.content
    task_id = [line for line in listed.content.splitlines() if line.startswith("- t_")][0].split(":")[0].replace("- ", "")

    updated = reg.dispatch(
        ToolCall(
            id="3",
            name="task_update",
            arguments={"task_id": task_id, "status": "in_progress", "notes": "Started"},
        )
    )
    assert "Updated" in updated.content

    listed2 = reg.dispatch(ToolCall(id="4", name="task_list", arguments={"status": "in_progress"}))
    assert task_id in listed2.content


def test_plan_slash_toggle_and_status(tmp_path):
    _, _, ext_ctx, prints = _ctx(tmp_path)
    ext = TaskPlannerExtension()
    ext.on_load(ext_ctx)

    assert ext.handle_slash("plan", "status", ext_ctx) is True
    assert any("Plan mode:" in p for p in prints)
    prints.clear()

    assert ext.handle_slash("plan", "on", ext_ctx) is True
    assert any("set to on" in p for p in prints)
    prints.clear()

    assert ext.handle_slash("plan", "off", ext_ctx) is True
    assert any("set to off" in p for p in prints)
