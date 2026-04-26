"""Tests for tau.core.rate_limit_tracker."""

import time

from tau.core.rate_limit_tracker import (
    RateLimitBucket,
    RateLimitState,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
    _fmt_count,
    _fmt_seconds,
)


class TestParseHeaders:
    def test_no_headers(self):
        assert parse_rate_limit_headers({}) is None

    def test_irrelevant_headers(self):
        assert parse_rate_limit_headers({"content-type": "application/json"}) is None

    def test_full_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-reset-requests": "30",
            "x-ratelimit-limit-requests-1h": "1000",
            "x-ratelimit-remaining-requests-1h": "900",
            "x-ratelimit-reset-requests-1h": "3600",
            "x-ratelimit-limit-tokens": "200000",
            "x-ratelimit-remaining-tokens": "180000",
            "x-ratelimit-reset-tokens": "30",
            "x-ratelimit-limit-tokens-1h": "2000000",
            "x-ratelimit-remaining-tokens-1h": "1800000",
            "x-ratelimit-reset-tokens-1h": "3600",
        }
        state = parse_rate_limit_headers(headers, provider="openai")
        assert state is not None
        assert state.has_data
        assert state.provider == "openai"
        assert state.requests_min.limit == 100
        assert state.requests_min.remaining == 95
        assert state.requests_hour.limit == 1000
        assert state.tokens_min.limit == 200000
        assert state.tokens_hour.remaining == 1800000

    def test_case_insensitive(self):
        headers = {
            "X-Ratelimit-Limit-Requests": "50",
            "X-Ratelimit-Remaining-Requests": "48",
        }
        state = parse_rate_limit_headers(headers)
        assert state is not None
        assert state.requests_min.limit == 50

    def test_partial_headers(self):
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
        }
        state = parse_rate_limit_headers(headers)
        assert state is not None
        assert state.requests_min.limit == 100
        assert state.tokens_min.limit == 0  # No token headers


class TestBucket:
    def test_used(self):
        b = RateLimitBucket(limit=100, remaining=95)
        assert b.used == 5

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=80)
        assert b.usage_pct == 20.0

    def test_usage_pct_zero_limit(self):
        b = RateLimitBucket(limit=0, remaining=0)
        assert b.usage_pct == 0.0

    def test_remaining_seconds_now(self):
        b = RateLimitBucket(reset_seconds=60.0, captured_at=time.time() - 10)
        assert 49.0 <= b.remaining_seconds_now <= 51.0


class TestFormatting:
    def test_fmt_count(self):
        assert _fmt_count(500) == "500"
        assert _fmt_count(1500) == "1.5K"
        assert _fmt_count(1_500_000) == "1.5M"

    def test_fmt_seconds(self):
        assert _fmt_seconds(30) == "30s"
        assert _fmt_seconds(90) == "1m 30s"
        assert _fmt_seconds(3660) == "1h 1m"
        assert _fmt_seconds(3600) == "1h"

    def test_display_no_data(self):
        state = RateLimitState()
        result = format_rate_limit_display(state)
        assert "No rate limit data" in result

    def test_display_with_data(self):
        state = RateLimitState(
            requests_min=RateLimitBucket(limit=100, remaining=95, captured_at=time.time()),
            captured_at=time.time(),
            provider="openai",
        )
        result = format_rate_limit_display(state)
        assert "Openai" in result
        assert "Requests/min" in result

    def test_compact_no_data(self):
        state = RateLimitState()
        assert "No rate limit data" in format_rate_limit_compact(state)

    def test_compact_with_data(self):
        state = RateLimitState(
            requests_min=RateLimitBucket(limit=100, remaining=50, captured_at=time.time()),
            captured_at=time.time(),
        )
        result = format_rate_limit_compact(state)
        assert "RPM: 50/100" in result

    def test_warning_at_80_pct(self):
        state = RateLimitState(
            requests_min=RateLimitBucket(
                limit=100, remaining=15, reset_seconds=30,
                captured_at=time.time(),
            ),
            captured_at=time.time(),
        )
        result = format_rate_limit_display(state)
        assert "⚠" in result
