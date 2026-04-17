import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state circuit breaker for external service calls.

    CLOSED → normal operation.
    OPEN   → failing fast after threshold failures; blocks calls.
    HALF_OPEN → recovery probe allowed after recovery_timeout elapses.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float = 0.0
        self._state = _State.CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> _State:
        if (
            self._state is _State.OPEN
            and time.monotonic() - self._opened_at >= self._recovery_timeout
        ):
            return _State.HALF_OPEN
        return self._state

    @property
    def state_name(self) -> str:
        return self.state.value

    async def call(self, coro):
        async with self._lock:
            current = self.state
            if current is _State.OPEN:
                raise RuntimeError(
                    f"Service '{self.name}' is temporarily unavailable. Please try again shortly."
                )

        try:
            result = await coro
        except Exception:
            async with self._lock:
                self._failures += 1
                if self._failures >= self._threshold and self._state is not _State.OPEN:
                    self._state = _State.OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit breaker OPEN: %s (failures=%d)",
                        self.name,
                        self._failures,
                    )
            raise

        async with self._lock:
            if self._state is not _State.CLOSED:
                logger.info("Circuit breaker CLOSED: %s (recovered)", self.name)
            self._failures = 0
            self._state = _State.CLOSED
        return result
