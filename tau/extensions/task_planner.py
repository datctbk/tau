"""task_planner — built-in task lifecycle and plan mode extension.

Adds first-class task primitives:
  - task_create
  - task_update
  - task_list

And slash commands:
  - /tasks
  - /plan on|off|status
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import ExtensionManifest, SlashCommand, ToolDefinition, ToolParameter

_TASK_STATUSES = ("todo", "in_progress", "blocked", "done", "cancelled")
_TASK_PRIORITIES = ("low", "medium", "high")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _TaskStore:
    def __init__(self, workspace_root: str) -> None:
        self._path = Path(workspace_root).resolve() / ".tau" / "tasks.json"

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {"plan_mode": False, "tasks": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"plan_mode": False, "tasks": []}
            tasks = data.get("tasks", [])
            if not isinstance(tasks, list):
                tasks = []
            return {
                "plan_mode": bool(data.get("plan_mode", False)),
                "tasks": tasks,
            }
        except Exception:
            return {"plan_mode": False, "tasks": []}

    def save(self, state: dict[str, Any]) -> None:
        self._ensure_parent()
        self._path.write_text(json.dumps(state, indent=2), encoding="utf-8")


class TaskPlannerExtension(Extension):
    manifest = ExtensionManifest(
        name="task_planner",
        version="0.1.0",
        description=(
            "Task lifecycle tools (create/update/list) with plan mode for long-running work."
        ),
        author="tau",
        system_prompt_fragment=(
            "When work has multiple steps, keep a task list using task_create, "
            "task_update, and task_list. Mark progress as you go."
        ),
    )

    def __init__(self) -> None:
        self._workspace_root = "."
        self._store = _TaskStore(self._workspace_root)
        self._state = {"plan_mode": False, "tasks": []}
        self._ext_context: ExtensionContext | None = None

    def on_load(self, context: ExtensionContext) -> None:
        self._ext_context = context
        workspace = "."
        if hasattr(context, "_agent_config") and context._agent_config:
            workspace = getattr(context._agent_config, "workspace_root", ".") or "."
        self._workspace_root = workspace
        self._store = _TaskStore(self._workspace_root)
        self._state = self._store.load()

    def before_turn(self, user_input: str) -> None:
        if not self._state.get("plan_mode", False):
            return
        open_tasks = [t for t in self._state.get("tasks", []) if t.get("status") not in {"done", "cancelled"}]
        if not open_tasks:
            return
        lines = [
            "Plan mode is enabled. Start with a short execution plan and keep task statuses updated.",
            "Open tasks:",
        ]
        for t in open_tasks[:8]:
            lines.append(f"- [{t.get('status', 'todo')}] {t.get('id', '?')}: {t.get('title', '')}")
        ctx = getattr(self, "_ext_context", None)
        if ctx is not None and hasattr(ctx, "_context"):
            try:
                ctx._context.inject_prompt_fragment(  # noqa: SLF001
                    "\n".join(lines), name="task_planner_state", priority=58
                )
            except Exception:
                pass

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="task_create",
                description="Create a tracked task for multi-step work.",
                parameters={
                    "title": ToolParameter(type="string", description="Short task title."),
                    "description": ToolParameter(type="string", description="Task details.", required=False),
                    "priority": ToolParameter(
                        type="string",
                        description="Task priority.",
                        enum=list(_TASK_PRIORITIES),
                        required=False,
                    ),
                },
                handler=self._handle_task_create,
            ),
            ToolDefinition(
                name="task_update",
                description="Update an existing tracked task.",
                parameters={
                    "task_id": ToolParameter(type="string", description="Task id, e.g. t_ab12cd34."),
                    "status": ToolParameter(
                        type="string",
                        description="New task status.",
                        enum=list(_TASK_STATUSES),
                        required=False,
                    ),
                    "title": ToolParameter(type="string", description="Updated title.", required=False),
                    "description": ToolParameter(type="string", description="Updated description.", required=False),
                    "notes": ToolParameter(type="string", description="Progress note to append.", required=False),
                    "priority": ToolParameter(
                        type="string",
                        description="Updated priority.",
                        enum=list(_TASK_PRIORITIES),
                        required=False,
                    ),
                },
                handler=self._handle_task_update,
            ),
            ToolDefinition(
                name="task_list",
                description="List tracked tasks with optional filtering.",
                parameters={
                    "status": ToolParameter(
                        type="string",
                        description="Filter by status.",
                        enum=list(_TASK_STATUSES),
                        required=False,
                    ),
                    "include_done": ToolParameter(
                        type="boolean",
                        description="Include done/cancelled tasks (default true).",
                        required=False,
                    ),
                },
                handler=self._handle_task_list,
            ),
        ]

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(name="tasks", description="Show tracked tasks.", usage="/tasks [status]"),
            SlashCommand(name="plan", description="Toggle plan mode.", usage="/plan on|off|status"),
        ]

    def handle_slash(self, command: str, args: str, context: ExtensionContext) -> bool:
        if command == "tasks":
            filt = args.strip() or None
            context.print(self._handle_task_list(status=filt, include_done=True))
            return True
        if command == "plan":
            arg = args.strip().lower()
            if arg in {"", "status"}:
                mode = "on" if self._state.get("plan_mode", False) else "off"
                context.print(f"[cyan]Plan mode:[/cyan] {mode}")
                return True
            if arg in {"on", "off"}:
                self._state["plan_mode"] = arg == "on"
                self._store.save(self._state)
                context.print(f"[green]Plan mode set to {arg}.[/green]")
                return True
            context.print("[dim]Usage: /plan on|off|status[/dim]")
            return True
        return False

    def _handle_task_create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
    ) -> str:
        if not title.strip():
            return "Error: title is required."
        if priority not in _TASK_PRIORITIES:
            priority = "medium"
        task_id = f"t_{uuid4().hex[:8]}"
        now = _now_iso()
        task = {
            "id": task_id,
            "title": title.strip(),
            "description": description.strip(),
            "status": "todo",
            "priority": priority,
            "created_at": now,
            "updated_at": now,
            "notes": [],
        }
        self._state.setdefault("tasks", []).append(task)
        self._store.save(self._state)
        return f"Created task {task_id}: {task['title']} (priority={priority}, status=todo)"

    def _handle_task_update(
        self,
        task_id: str,
        status: str | None = None,
        title: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        priority: str | None = None,
    ) -> str:
        tasks = self._state.get("tasks", [])
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if task is None:
            return f"Error: task not found: {task_id}"
        changed: list[str] = []
        if status is not None:
            if status not in _TASK_STATUSES:
                return f"Error: invalid status {status!r}. Allowed: {', '.join(_TASK_STATUSES)}"
            task["status"] = status
            changed.append(f"status={status}")
        if title is not None and title.strip():
            task["title"] = title.strip()
            changed.append("title")
        if description is not None:
            task["description"] = description.strip()
            changed.append("description")
        if priority is not None:
            if priority not in _TASK_PRIORITIES:
                return f"Error: invalid priority {priority!r}. Allowed: {', '.join(_TASK_PRIORITIES)}"
            task["priority"] = priority
            changed.append(f"priority={priority}")
        if notes is not None and notes.strip():
            task.setdefault("notes", []).append({"at": _now_iso(), "text": notes.strip()})
            changed.append("notes")
        task["updated_at"] = _now_iso()
        self._store.save(self._state)
        return f"Updated {task_id}: {', '.join(changed) if changed else 'no fields changed'}"

    def _handle_task_list(
        self,
        status: str | None = None,
        include_done: bool = True,
    ) -> str:
        tasks = list(self._state.get("tasks", []))
        if status:
            if status not in _TASK_STATUSES:
                return f"Error: invalid status {status!r}. Allowed: {', '.join(_TASK_STATUSES)}"
            tasks = [t for t in tasks if t.get("status") == status]
        elif not include_done:
            tasks = [t for t in tasks if t.get("status") not in {"done", "cancelled"}]

        if not tasks:
            return "No tasks."
        lines = [f"Tasks ({len(tasks)}):"]
        order = {s: i for i, s in enumerate(_TASK_STATUSES)}
        tasks.sort(key=lambda t: (order.get(t.get("status", "todo"), 99), t.get("id", "")))
        for t in tasks:
            lines.append(
                f"- {t.get('id')}: [{t.get('status')}] {t.get('title')} "
                f"(priority={t.get('priority', 'medium')})"
            )
        return "\n".join(lines)


EXTENSION = TaskPlannerExtension()
