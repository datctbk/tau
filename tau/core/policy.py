"""Policy hook scaffold for tool-call risk controls.

Phase A goal: provide a stable hook point without changing default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Protocol

from tau.core.types import ToolCall

if TYPE_CHECKING:
    from tau.core.agent import Agent


@dataclass
class PolicyDecision:
    allow: bool
    requires_approval: bool = False
    reason: str | None = None
    risk: str = "low"


class ToolPolicyHook(Protocol):
    def before_tool_call(self, *, agent: "Agent", call: ToolCall) -> PolicyDecision: ...


class PolicyProfileEvaluator(Protocol):
    def decide(self, *, profile: str, call: ToolCall) -> PolicyDecision: ...


class DefaultPolicyProfileEvaluator:
    """Built-in profile evaluator kept inside core (no extension coupling)."""

    @staticmethod
    def _is_destructive_shell(command: str) -> bool:
        risky = [
            r"\brm\s+-rf\b",
            r"\bmkfs\b",
            r"\bdd\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bhalt\b",
            r"\bchmod\s+777\b",
            r"\bchown\s+-R\b",
            r"\bcurl\b.*\|\s*sh\b",
            r"\bwget\b.*\|\s*sh\b",
            r">\s*/dev/sd",
        ]
        c = command.lower()
        return any(re.search(p, c) for p in risky)

    def _classify_risk(self, call: ToolCall) -> str:
        name = call.name
        if name in {"read_file", "list_dir", "search_files", "grep", "find", "ls", "task_events"}:
            return "low"
        if name in {"write_file", "edit_file", "task_update", "task_stop", "task_create"}:
            return "medium"
        if name == "run_bash":
            command = str(call.arguments.get("command", ""))
            return "high" if self._is_destructive_shell(command) else "medium"
        if name in {"web_search", "web_fetch", "agent"}:
            return "high"
        return "medium"

    def decide(self, *, profile: str, call: ToolCall) -> PolicyDecision:
        risk = self._classify_risk(call)

        if profile == "dev":
            return PolicyDecision(allow=True, requires_approval=False, risk=risk)

        if profile == "strict":
            if risk in {"high", "medium"}:
                return PolicyDecision(
                    allow=True,
                    requires_approval=True,
                    risk=risk,
                    reason=f"Approval required by strict policy: {call.name} ({risk})",
                )
            return PolicyDecision(allow=True, requires_approval=False, risk=risk)

        # balanced profile
        if risk in {"high", "medium"}:
            return PolicyDecision(
                allow=True,
                requires_approval=True,
                risk=risk,
                reason=f"Approval required by balanced policy: {call.name} ({risk})",
            )
        return PolicyDecision(allow=True, requires_approval=False, risk=risk)


_policy_profile_evaluator: PolicyProfileEvaluator | None = None


def register_policy_profile_evaluator(evaluator: PolicyProfileEvaluator | None) -> None:
    """Register/override the policy profile evaluator used by core policy hook."""
    global _policy_profile_evaluator
    _policy_profile_evaluator = evaluator


def clear_policy_profile_evaluator() -> None:
    """Clear any externally registered evaluator and fall back to core default."""
    register_policy_profile_evaluator(None)


class DefaultToolPolicyHook:
    """Default policy scaffold.

    Current behavior is permissive to avoid breaking existing workflows.
    Future phases can tighten checks per profile.
    """

    def __init__(self, *, profile: str = "balanced") -> None:
        self._profile = profile
        self._evaluator = _policy_profile_evaluator or DefaultPolicyProfileEvaluator()

    @staticmethod
    def _is_preapproved_upstream(call: ToolCall) -> bool:
        """Avoid duplicate approval prompts when caller already gated risk.

        Any tool may pass ``approved_risky_actions=true`` when explicit user
        approval has already been captured upstream.
        """
        return bool(call.arguments.get("approved_risky_actions", False))

    def before_tool_call(self, *, agent: "Agent", call: ToolCall) -> PolicyDecision:
        _ = agent
        decision = self._evaluator.decide(profile=self._profile, call=call)
        if decision.requires_approval and self._is_preapproved_upstream(call):
            return PolicyDecision(
                allow=decision.allow,
                requires_approval=False,
                risk=decision.risk,
                reason="Approval already confirmed by upstream workflow gate.",
            )
        return decision
