"""Steering channel — thread-safe steer + follow-up queue for the agent loop."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field


@dataclass
class _Steer:
    """A mid-stream steering injection."""
    message: str


@dataclass
class _Enqueue:
    """A follow-up prompt added to the back of the queue."""
    message: str


class SteeringChannel:
    """
    Shared communication channel between the REPL input thread and the agent loop.

    Two independent mechanisms:

    1. **Steer** – interrupt the currently-streaming response and immediately
       start a new turn with a replacement user message.
       Written by: REPL thread (bare input while streaming).
       Read by:    Agent._stream(), checked after each TextDelta.

    2. **Follow-up queue** – a FIFO of prompts that are consumed one-by-one
       after each turn completes (like a playlist).
       Written by: REPL thread (/queue <msg>).
       Read by:    Agent.run() after TurnComplete.
    """

    def __init__(self) -> None:
        # At most one pending steer at a time — older ones are overwritten.
        self._steer: str | None = None
        self._steer_lock = threading.Lock()

        # Ordered queue of follow-up prompts.
        self._followups: queue.SimpleQueue[str] = queue.SimpleQueue()

    # ------------------------------------------------------------------
    # Steer (interrupt current stream)
    # ------------------------------------------------------------------

    def steer(self, message: str) -> None:
        """Called from the input thread: request a mid-stream steer."""
        with self._steer_lock:
            self._steer = message

    def consume_steer(self) -> str | None:
        """Called from the agent thread: take the pending steer (if any)."""
        with self._steer_lock:
            msg = self._steer
            self._steer = None
            return msg

    def has_steer(self) -> bool:
        with self._steer_lock:
            return self._steer is not None

    def clear_steer(self) -> None:
        with self._steer_lock:
            self._steer = None

    # ------------------------------------------------------------------
    # Follow-up queue
    # ------------------------------------------------------------------

    def enqueue(self, message: str) -> None:
        """Add a follow-up prompt to the back of the queue."""
        self._followups.put(message)

    def dequeue(self) -> str | None:
        """Pop the next follow-up prompt, or None if the queue is empty."""
        try:
            return self._followups.get_nowait()
        except queue.Empty:
            return None

    def queue_size(self) -> int:
        return self._followups.qsize()

    def drain(self) -> list[str]:
        """Remove and return all queued follow-up prompts."""
        items: list[str] = []
        while True:
            try:
                items.append(self._followups.get_nowait())
            except queue.Empty:
                break
        return items
