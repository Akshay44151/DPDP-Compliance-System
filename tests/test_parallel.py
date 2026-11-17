import threading

import pytest

from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)
from orchestrator.parallel import TaskOutcome, run_in_parallel


def test_every_result_is_returned_in_input_order():
    outcomes = run_in_parallel({"a": lambda: 1, "b": lambda: 2, "c": lambda: 3})

    assert list(outcomes) == ["a", "b", "c"]
    assert [o.value for o in outcomes.values()] == [1, 2, 3]
    assert all(o.ok for o in outcomes.values())


def test_a_failing_task_is_captured_and_others_still_finish():
    def boom():
        raise RuntimeError("agent exploded")

    outcomes = run_in_parallel({"good": lambda: "fine", "bad": boom})

    assert outcomes["good"] == TaskOutcome(name="good", ok=True, value="fine")
    assert outcomes["bad"].ok is False
    assert outcomes["bad"].value is None
    assert "RuntimeError" in outcomes["bad"].error
    assert "agent exploded" in outcomes["bad"].error


def test_a_very_long_error_message_is_shortened():
    def boom():
        raise ValueError("x" * 1000)

    outcomes = run_in_parallel({"bad": boom})

    assert len(outcomes["bad"].error) < 400


def test_tasks_really_run_at_the_same_time():
    # Each task waits for the other. If they ran one after the other,
    # the first would time out and the barrier would break.
    barrier = threading.Barrier(2, timeout=2)

    def meet():
        barrier.wait()
        return "met"

    outcomes = run_in_parallel({"agent1": meet, "agent2": meet})

    assert all(o.ok for o in outcomes.values())


def test_no_tasks_gives_an_empty_result():
    assert run_in_parallel({}) == {}


def test_a_single_worker_still_runs_every_task():
    outcomes = run_in_parallel({"a": lambda: 1, "b": lambda: 2}, max_workers=1)

    assert [o.value for o in outcomes.values()] == [1, 2]


def test_zero_workers_is_rejected():
    with pytest.raises(ValueError):
        run_in_parallel({"a": lambda: 1}, max_workers=0)


def test_concurrent_logging_to_one_state_keeps_every_entry():
    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-1",
            policy_documents=[PolicyDocument(name="p.txt", text="text")],
        )
    )

    def make_task(i: int):
        return lambda: state.log("step", StepStatus.COMPLETED, target=f"t{i}")

    run_in_parallel({f"t{i}": make_task(i) for i in range(50)})

    assert len(state.step_log) == 50