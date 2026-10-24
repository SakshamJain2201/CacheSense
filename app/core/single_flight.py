"""In-flight request coalescing (single-flight) to eliminate thundering herd on cache misses."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class SingleFlight:
    """
    Ensures that for concurrent identical cache misses (matching partition_key + prompt_hash),
    only one upstream call is initiated (the leader), while concurrent identical callers (followers)
    await and share the leader's outcome.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._in_flight: dict[str, asyncio.Future[Any]] = {}

    async def execute(
        self,
        key: str,
        fn: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        """
        Execute `fn` if this is the first in-flight request for `key` (leader),
        or wait for the in-flight request to finish (follower).
        Returns (result, is_leader).
        """
        async with self._lock:
            if key in self._in_flight:
                future = self._in_flight[key]
                is_leader = False
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                is_leader = True

        if not is_leader:
            result = await future
            return result, False

        # Leader executes fn
        try:
            result = await fn()
            future.set_result(result)
            return result, True
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)
