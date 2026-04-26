from __future__ import annotations

import ast
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parents[1] / "tau" / "core"

FORBIDDEN_IMPORT_PREFIXES = (
    "tau_assistant",
    "tau_memory",
    "tau_web",
    "tau_agents",
    "tau_gateway",
)

FORBIDDEN_REPO_PATH_LITERALS = (
    "tau-assistant",
    "tau-memory",
    "tau-web",
    "tau-agents",
    "tau-gateway",
)


def _iter_core_py_files() -> list[Path]:
    return sorted(
        p for p in CORE_DIR.rglob("*.py")
        if p.is_file()
    )


def test_core_does_not_import_extension_repos_or_modules() -> None:
    violations: list[str] = []
    for file_path in _iter_core_py_files():
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{file_path}: import {name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{file_path}: from {node.module} import ...")

    assert not violations, (
        "Core/extension boundary violated. Move these imports to extensions:\n"
        + "\n".join(violations)
    )


def test_core_does_not_probe_extension_repo_paths_directly() -> None:
    violations: list[str] = []
    for file_path in _iter_core_py_files():
        text = file_path.read_text(encoding="utf-8")
        for token in FORBIDDEN_REPO_PATH_LITERALS:
            if token in text:
                violations.append(f"{file_path}: contains '{token}'")

    assert not violations, (
        "Core must not hard-code extension repo paths. Found:\n"
        + "\n".join(violations)
    )

