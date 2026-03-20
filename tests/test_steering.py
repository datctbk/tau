"""Tests for streaming steering & follow-up queue (Feature #3)."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from tau.core.agent import Agent
from tau.core.context import ContextManager
from tau.core.session import Session, SessionManager
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import (
    AgentConfig,
    ErrorEvent,
    Message,
    ProviderResponse,
    SteerEvent,
    TextChunk,
    TextDelta,
    TokenUsage,
    TurnComplete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> AgentConfig:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        max_tokens=8192,
        max_turns=10,
        system_prompt="sys",
        retry_enabled=False,      # keep retry out of the picture
        compaction_enabled=False, # keep compaction out of the picture
    )
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_agent(
    provider_responses: list,
    steering: SteeringChannel | None = None,
    cfg: AgentConfig | None = None,
) -> Agent:
    config = cfg or _cfg()
    context = ContextManager(config)
    registry = ToolRegistry()
    provider = MagicMock()
    provider.chat.side_effect = provider_responses

    sm = MagicMock(spec=SessionManager)
    sm.save.return_value = None
    sm.append_compaction.return_value = None

    session = MagicMock(spec=Session)
    session.id = "test"
    session.messages = []
    session.compactions = []

    return Agent(
        config=config,
        provider=provider,
        registry=registry,
        context=context,
        session=session,
        session_manager=sm,
        steering=steering,
    )


def _ok(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _streaming_response(words: list[str], stop_reason: str = "end_turn"):
    """Generator that yields one TextDelta per word, then a ProviderResponse."""
    def _gen():
        for w in words:
            yield TextDelta(text=w)
        yield ProviderResponse(
            content="".join(words),
            tool_calls=[],
            stop_reason=stop_reason,
            usage=TokenUsage(input_tokens=10, output_tokens=len(words)),
        )
    return _gen()


# ===========================================================================
# SteeringChannel unit tests
# ===========================================================================

class TestSteeringChannelSteer:
    def test_initial_steer_is_none(self):
        ch = SteeringChannel()
        assert ch.consume_steer() is None

    def test_steer_then_consume(self):
        ch = SteeringChannel()
        ch.steer("go left")
        assert ch.consume_steer() == "go left"

    def test_consume_clears_steer(self):
        ch = SteeringChannel()
        ch.steer("msg")
        ch.consume_steer()
        assert ch.consume_steer() is None

    def test_steer_overwrites_previous(self):
        ch = SteeringChannel()
        ch.steer("first")
        ch.steer("second")
        assert ch.consume_steer() == "second"

    def test_has_steer_true_when_pending(self):
        ch = SteeringChannel()
        ch.steer("x")
        assert ch.has_steer() is True

    def test_has_steer_false_when_empty(self):
        ch = SteeringChannel()
        assert ch.has_steer() is False

    def test_has_steer_false_after_consume(self):
        ch = SteeringChannel()
        ch.steer("x")
        ch.consume_steer()
        assert ch.has_steer() is False

    def test_clear_steer_removes_pending(self):
        ch = SteeringChannel()
        ch.steer("x")
        ch.clear_steer()
        assert ch.has_steer() is False
        assert ch.consume_steer() is None

    def test_thread_safe_steer(self):
        """Multiple threads writing steers — final consume returns one of them."""
        ch = SteeringChannel()
        results = []

        def writer(msg):
            ch.steer(msg)

        threads = [threading.Thread(target=writer, args=(f"msg{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = ch.consume_steer()
        assert result is not None
        assert result.startswith("msg")


class TestSteeringChannelQueue:
    def test_empty_queue_returns_none(self):
        ch = SteeringChannel()
        assert ch.dequeue() is None

    def test_enqueue_then_dequeue(self):
        ch = SteeringChannel()
        ch.enqueue("first")
        assert ch.dequeue() == "first"

    def test_dequeue_is_fifo(self):
        ch = SteeringChannel()
        ch.enqueue("a")
        ch.enqueue("b")
        ch.enqueue("c")
        assert ch.dequeue() == "a"
        assert ch.dequeue() == "b"
        assert ch.dequeue() == "c"

    def test_dequeue_returns_none_when_exhausted(self):
        ch = SteeringChannel()
        ch.enqueue("x")
        ch.dequeue()
        assert ch.dequeue() is None

    def test_queue_size(self):
        ch = SteeringChannel()
        assert ch.queue_size() == 0
        ch.enqueue("a")
        ch.enqueue("b")
        assert ch.queue_size() == 2
        ch.dequeue()
        assert ch.queue_size() == 1

    def test_drain_empties_queue(self):
        ch = SteeringChannel()
        ch.enqueue("a")
        ch.enqueue("b")
        ch.enqueue("c")
        drained = ch.drain()
        assert drained == ["a", "b", "c"]
        assert ch.queue_size() == 0

    def test_drain_empty_returns_empty_list(self):
        ch = SteeringChannel()
        assert ch.drain() == []

    def test_thread_safe_enqueue_dequeue(self):
        """Concurrent enqueues from multiple threads; all items retrievable."""
        ch = SteeringChannel()
        n = 50

        def enqueuer(i):
            ch.enqueue(f"item{i}")

        threads = [threading.Thread(target=enqueuer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        items = ch.drain()
        assert len(items) == n

    def test_steer_and_queue_independent(self):
        """Steer slot and follow-up queue are independent."""
        ch = SteeringChannel()
        ch.steer("steer me")
        ch.enqueue("follow up 1")
        ch.enqueue("follow up 2")

        assert ch.consume_steer() == "steer me"
        assert ch.dequeue() == "follow up 1"
        assert ch.dequeue() == "follow up 2"
        assert ch.consume_steer() is None


# ===========================================================================
# Agent — no steering channel attached (baseline: nothing breaks)
# ===========================================================================

class TestAgentNoSteering:
    def test_plain_response_no_steer_event(self):
        agent = _make_agent([_ok("hello")])
        events = list(agent.run("hi"))
        assert not any(isinstance(e, SteerEvent) for e in events)
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_streaming_response_no_steer_event(self):
        agent = _make_agent([_streaming_response(["a", "b", "c"])])
        events = list(agent.run("go"))
        assert not any(isinstance(e, SteerEvent) for e in events)
        deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(deltas) == 3


# ===========================================================================
# Agent — mid-stream steer
# ===========================================================================

class TestAgentMidStreamSteer:

    def _agent_with_steer_after_n_deltas(
        self,
        steer_msg: str,
        steer_after: int,
        streaming_words: list[str],
        final_response: ProviderResponse,
    ) -> tuple[Agent, SteeringChannel]:
        """
        Build an agent where the SteeringChannel fires after *steer_after* deltas.
        The streaming response yields from *streaming_words*, and after the steer
        the provider returns *final_response*.
        """
        ch = SteeringChannel()

        delta_count = 0
        original_consume = ch.consume_steer

        def patched_consume():
            nonlocal delta_count
            delta_count += 1
            if delta_count > steer_after:
                return steer_msg
            return None

        ch.consume_steer = patched_consume  # type: ignore[method-assign]

        agent = _make_agent(
            provider_responses=[
                _streaming_response(streaming_words),
                final_response,
            ],
            steering=ch,
        )
        return agent, ch

    def test_steer_emits_steer_event(self):
        agent, _ = self._agent_with_steer_after_n_deltas(
            steer_msg="new direction",
            steer_after=1,
            streaming_words=["word1", "word2", "word3"],
            final_response=_ok("steered answer"),
        )
        events = list(agent.run("original"))
        assert any(isinstance(e, SteerEvent) for e in events)

    def test_steer_event_carries_new_input(self):
        agent, _ = self._agent_with_steer_after_n_deltas(
            steer_msg="go right",
            steer_after=1,
            streaming_words=["w1", "w2"],
            final_response=_ok("ok"),
        )
        events = list(agent.run("original"))
        steer_events = [e for e in events if isinstance(e, SteerEvent)]
        assert steer_events[0].new_input == "go right"

    def test_steer_continues_to_final_answer(self):
        agent, _ = self._agent_with_steer_after_n_deltas(
            steer_msg="new topic",
            steer_after=1,
            streaming_words=["a", "b", "c"],
            final_response=_ok("final"),
        )
        events = list(agent.run("start"))
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_steer_adds_new_user_message_to_context(self):
        cfg = _cfg()
        ch = SteeringChannel()

        # Inject steer after first delta
        call_count = 0
        original = ch.consume_steer

        def patched():
            nonlocal call_count
            call_count += 1
            return "steered input" if call_count > 1 else None

        ch.consume_steer = patched  # type: ignore[method-assign]

        context = ContextManager(cfg)
        registry = ToolRegistry()
        provider = MagicMock()
        provider.chat.side_effect = [
            _streaming_response(["tok1", "tok2"]),
            _ok("answer"),
        ]
        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None
        session = MagicMock(spec=Session)
        session.id = "t"
        session.messages = []
        session.compactions = []

        agent = Agent(
            config=cfg, provider=provider, registry=registry,
            context=context, session=session, session_manager=sm,
            steering=ch,
        )
        list(agent.run("original input"))

        user_msgs = [m for m in context.get_messages() if m.role == "user"]
        contents = [m.content for m in user_msgs]
        assert "steered input" in contents

    def test_steer_before_any_delta_still_works(self):
        """Steer fires on the very first delta check — no deltas emitted yet."""
        ch = SteeringChannel()
        ch.steer("immediate steer")

        agent = _make_agent(
            provider_responses=[
                _streaming_response(["tok1", "tok2", "tok3"]),
                _ok("post-steer answer"),
            ],
            steering=ch,
        )
        events = list(agent.run("go"))
        assert any(isinstance(e, SteerEvent) for e in events)
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_no_steer_event_when_stream_completes_normally(self):
        """If the steer never fires, no SteerEvent is emitted."""
        ch = SteeringChannel()   # nothing ever calls .steer()
        agent = _make_agent(
            provider_responses=[_streaming_response(["a", "b"])],
            steering=ch,
        )
        events = list(agent.run("hi"))
        assert not any(isinstance(e, SteerEvent) for e in events)
        assert any(isinstance(e, TurnComplete) for e in events)

    def test_steer_resets_turn_counter(self):
        """After a steer, turns should reset so the new request gets a full budget."""
        cfg = _cfg(max_turns=2)
        ch = SteeringChannel()

        call_n = 0

        def patched_consume():
            nonlocal call_n
            call_n += 1
            # Fire steer on the first streaming chunk of the first turn
            return "redirect" if call_n == 1 else None

        ch.consume_steer = patched_consume  # type: ignore[method-assign]

        agent = _make_agent(
            provider_responses=[
                _streaming_response(["tok"]),   # first turn — gets steered
                _ok("final"),                   # second turn after steer
            ],
            steering=ch,
            cfg=cfg,
        )
        events = list(agent.run("start"))
        # Should complete successfully, not hit max_turns error
        assert any(isinstance(e, TurnComplete) for e in events)
        assert not any(isinstance(e, ErrorEvent) for e in events)

    def test_blocking_provider_no_steer_check(self):
        """Blocking (non-streaming) responses are never interrupted by a steer."""
        ch = SteeringChannel()
        ch.steer("try to steer")   # steer is set but can't interrupt a blocking call

        agent = _make_agent(
            provider_responses=[_ok("blocking answer")],
            steering=ch,
        )
        events = list(agent.run("go"))
        # The steer was pending but couldn't fire — no SteerEvent
        assert not any(isinstance(e, SteerEvent) for e in events)
        assert any(isinstance(e, TurnComplete) for e in events)


# ===========================================================================
# Agent — follow-up queue
# ===========================================================================

class TestAgentFollowUpQueue:

    def test_single_followup_processed(self):
        ch = SteeringChannel()
        ch.enqueue("follow-up question")

        agent = _make_agent(
            provider_responses=[
                _ok("first answer"),
                _ok("follow-up answer"),
            ],
            steering=ch,
        )
        events = list(agent.run("first question"))

        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 2

    def test_multiple_followups_all_processed(self):
        ch = SteeringChannel()
        ch.enqueue("q2")
        ch.enqueue("q3")
        ch.enqueue("q4")

        agent = _make_agent(
            provider_responses=[_ok(f"answer {i}") for i in range(4)],
            steering=ch,
        )
        events = list(agent.run("q1"))

        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 4

    def test_followups_consumed_in_order(self):
        ch = SteeringChannel()
        ch.enqueue("second")
        ch.enqueue("third")

        received_inputs: list[str] = []

        cfg = _cfg()
        context = ContextManager(cfg)
        registry = ToolRegistry()

        provider = MagicMock()
        provider.chat.side_effect = [_ok("a"), _ok("b"), _ok("c")]

        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None
        session = MagicMock(spec=Session)
        session.id = "t"
        session.messages = []
        session.compactions = []

        agent = Agent(
            config=cfg, provider=provider, registry=registry,
            context=context, session=session, session_manager=sm,
            steering=ch,
        )
        list(agent.run("first"))

        user_msgs = [m for m in context.get_messages() if m.role == "user"]
        contents = [m.content for m in user_msgs]
        assert contents == ["first", "second", "third"]

    def test_no_followup_single_turn(self):
        ch = SteeringChannel()   # empty queue
        agent = _make_agent([_ok("done")], steering=ch)
        events = list(agent.run("hi"))
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1

    def test_followup_queue_empty_after_processing(self):
        ch = SteeringChannel()
        ch.enqueue("follow1")
        ch.enqueue("follow2")

        agent = _make_agent(
            provider_responses=[_ok("a"), _ok("b"), _ok("c")],
            steering=ch,
        )
        list(agent.run("start"))

        assert ch.queue_size() == 0

    def test_followup_without_steering_channel(self):
        """Agent with no SteeringChannel attached processes only the initial input."""
        agent = _make_agent([_ok("only one turn")])
        events = list(agent.run("go"))
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1

    def test_followup_after_streaming_turn(self):
        ch = SteeringChannel()
        ch.enqueue("follow-up")

        agent = _make_agent(
            provider_responses=[
                _streaming_response(["streamed ", "response"]),
                _ok("follow-up answer"),
            ],
            steering=ch,
        )
        events = list(agent.run("first"))
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 2

    def test_enqueue_during_run_is_consumed(self):
        """
        A follow-up enqueued while the first turn is running is picked up
        before run() returns.
        """
        ch = SteeringChannel()

        turn_count = 0
        original_ok = _ok

        responses = []

        def side_effect(*args, **kwargs):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 1:
                # Enqueue follow-up mid-flight (simulates REPL thread)
                ch.enqueue("late follow-up")
            if turn_count <= 2:
                return _ok(f"answer {turn_count}")
            return _ok("extra")

        cfg = _cfg()
        context = ContextManager(cfg)
        registry = ToolRegistry()
        provider = MagicMock()
        provider.chat.side_effect = side_effect

        sm = MagicMock(spec=SessionManager)
        sm.save.return_value = None
        sm.append_compaction.return_value = None
        session = MagicMock(spec=Session)
        session.id = "t"
        session.messages = []
        session.compactions = []

        agent = Agent(
            config=cfg, provider=provider, registry=registry,
            context=context, session=session, session_manager=sm,
            steering=ch,
        )
        events = list(agent.run("first"))
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 2


# ===========================================================================
# Agent — steer + follow-up combined
# ===========================================================================

class TestAgentSteerAndQueue:

    def test_steer_drains_pending_queue_is_still_processed(self):
        """
        Queue items enqueued before a steer should still be consumed after
        the steered turn completes.
        """
        ch = SteeringChannel()
        ch.enqueue("queued prompt")

        # Fire the steer on the first streaming delta
        call_n = 0

        def patched_consume():
            nonlocal call_n
            call_n += 1
            return "steer!" if call_n == 1 else None

        ch.consume_steer = patched_consume  # type: ignore[method-assign]

        agent = _make_agent(
            provider_responses=[
                _streaming_response(["tok"]),   # original turn → steered
                _ok("steered answer"),           # steered turn
                _ok("queued answer"),            # queued follow-up
            ],
            steering=ch,
        )
        events = list(agent.run("original"))

        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        steer_events = [e for e in events if isinstance(e, SteerEvent)]
        assert steer_events
        assert len(turn_completes) == 2   # steered + queued

    def test_steer_event_before_turn_complete(self):
        """SteerEvent is always emitted before the subsequent TurnComplete."""
        ch = SteeringChannel()

        call_n = 0

        def patched_consume():
            nonlocal call_n
            call_n += 1
            return "steer!" if call_n == 1 else None

        ch.consume_steer = patched_consume  # type: ignore[method-assign]

        agent = _make_agent(
            provider_responses=[
                _streaming_response(["tok"]),
                _ok("ok"),
            ],
            steering=ch,
        )
        events = list(agent.run("go"))
        steer_idx = next(i for i, e in enumerate(events) if isinstance(e, SteerEvent))
        turn_idx = next(i for i, e in enumerate(events) if isinstance(e, TurnComplete))
        assert steer_idx < turn_idx

    def test_only_one_steer_consumed_per_check(self):
        """Even if steer() is called many times, consume_steer() returns one message."""
        ch = SteeringChannel()
        for i in range(10):
            ch.steer(f"steer {i}")
        result = ch.consume_steer()
        assert result is not None
        # After consuming, it's gone
        assert ch.consume_steer() is None

    def test_queue_not_consumed_when_no_steering_channel(self):
        """Without a SteeringChannel, agent never tries to drain a queue."""
        agent = _make_agent([_ok("single turn")], steering=None)
        events = list(agent.run("go"))
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1
