from __future__ import annotations

import time

from gqmr.jobs import ProcessJob


def test_spawn_process_job_result() -> None:
    with ProcessJob("gqmr.jobs.fixtures:add", args=(2, 3)) as job:
        assert job.result(timeout=5.0) == 5


def test_spawn_process_job_cancels_within_two_seconds() -> None:
    job = ProcessJob("gqmr.jobs.fixtures:wait_forever").start()
    started = time.monotonic()
    job.cancel(grace_seconds=0.1)
    elapsed = time.monotonic() - started

    assert not job.running
    assert elapsed < 2.0
    job.close()
