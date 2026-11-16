"""Run independent tasks at the same time (Phase A, step 5).

Used for stages that do not depend on each other, such as Agent 1
(policy extraction) and Agent 2 (code analysis). Each task runs in its
own thread. A task that raises an exception is captured and reported,
and never stops the other tasks.

Thread safety: tasks may write to the same AssessmentState. Appending to
a Python list is atomic, so concurrent log entries are never lost or
corrupted, but the order of entries from different tasks can vary.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskOutcome:
    """What happened to one task."""

    name: str
    ok: bool
    value: Any = None
    error: str = ""


def _run_one(name: str, task: Callable[[], Any]) -> TaskOutcome:
    try:
        return TaskOutcome(name=name, ok=True, value=task())
    except Exception as exc:
        return TaskOutcome(
            name=name,
            ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:300]}",
        )


def run_in_parallel(
    tasks: dict[str, Callable[[], Any]],
    *,
    max_workers: int | None = None,
) -> dict[str, TaskOutcome]:
    """Run every task at the same time and return one outcome per task.

    The result keeps the order of the input dictionary. By default every
    task gets its own thread.
    """
    if not tasks:
        return {}

    workers = len(tasks) if max_workers is None else max_workers

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            name: pool.submit(_run_one, name, task)
            for name, task in tasks.items()
        }
        return {name: future.result() for name, future in futures.items()}