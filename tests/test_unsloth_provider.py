from __future__ import annotations

from tau.providers.unsloth_provider import _estimate_prompt_tokens_from_messages, _estimate_tokens


def test_unsloth_fallback_token_estimators_are_nonzero_for_text():
    messages = [
        {"role": "system", "content": "You are tau."},
        {"role": "user", "content": "Please create a daily brief at 7 AM."},
    ]
    assert _estimate_prompt_tokens_from_messages(messages) > 0
    assert _estimate_tokens("Short assistant response.") > 0

