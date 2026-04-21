"""Policy hook scaffold for tool-call risk controls.

Phase A goal: provide a stable hook point without changing default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
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


class _AllowAllEvaluator:
    """Minimal fallback: keep core safe and generic when no external evaluator is present."""

    def decide(self, *, profile: str, call: ToolCall) -> PolicyDecision:
        _ = profile
        _ = call
        return PolicyDecision(allow=True, requires_approval=False, risk="low")


def _load_external_evaluator() -> PolicyProfileEvaluator:
    """Load profile-specific policy evaluator from tau-assistant extension if present."""
    try:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "tau-assistant" / "policy_profiles.py",
        ]
        policy_file = next((p for p in candidates if p.is_file()), None)
        if policy_file is None:
            return _AllowAllEvaluator()

        spec = importlib.util.spec_from_file_location("_tau_assistant_policy_profiles", str(policy_file))
        if spec is None or spec.loader is None:
            return _AllowAllEvaluator()
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        evaluator_cls = getattr(mod, "DefaultPolicyProfileEvaluator", None)
        if evaluator_cls is None:
            return _AllowAllEvaluator()
        return evaluator_cls()
    except Exception:  # noqa: BLE001
        return _AllowAllEvaluator()


class DefaultToolPolicyHook:
    """Default policy scaffold.

    Current behavior is permissive to avoid breaking existing workflows.
    Future phases can tighten checks per profile.
    """

    def __init__(self, *, profile: str = "balanced") -> None:
        self._profile = profile
        self._evaluator = _load_external_evaluator()

    @staticmethod
    def _is_preapproved_by_assistant(call: ToolCall) -> bool:
        """Avoid duplicate approval prompts when assistant already gated risk.

        tau-assistant's workflow gate uses the explicit ``approved_risky_actions``
        parameter. If it is true, core policy should not prompt again for the same
        user decision.
        """
        if call.name != "assistant_workflow_run":
            return False
        return bool(call.arguments.get("approved_risky_actions", False))

    def before_tool_call(self, *, agent: "Agent", call: ToolCall) -> PolicyDecision:
        _ = agent
        decision = self._evaluator.decide(profile=self._profile, call=call)
        if decision.requires_approval and self._is_preapproved_by_assistant(call):
            return PolicyDecision(
                allow=decision.allow,
                requires_approval=False,
                risk=decision.risk,
                reason="Approval already confirmed by assistant workflow gate.",
            )
        return decision
