from event_automation import EventProcessor


def flaky(fail_times):
    state = {"left": fail_times}
    def handler(payload):
        if state["left"] > 0:
            state["left"] -= 1
            raise RuntimeError("transient")
        return f"done:{payload['x']}"
    return handler


def make(fail_times=0, max_retries=3):
    sleeps = []
    p = EventProcessor({"job": flaky(fail_times)}, max_retries=max_retries,
                       sleep=sleeps.append)
    return p, sleeps


def ev(i, type_="job"):
    return {"id": f"e{i}", "type": type_, "payload": {"x": i}}


def test_success_and_recorded():
    p, _ = make()
    assert p.process(ev(1)) == {"status": "ok", "result": "done:1"}
    assert p.processed == {"e1": "done:1"}


def test_idempotent_duplicate_skipped():
    p, _ = make()
    p.process(ev(1))
    calls_before = dict(p.processed)
    assert p.process(ev(1)) == {"status": "duplicate"}
    assert p.processed == calls_before


def test_retry_with_backoff_then_success():
    p, sleeps = make(fail_times=2)
    result = p.process(ev(1))
    assert result["status"] == "ok"
    assert sleeps == [1, 2]


def test_dead_letter_after_exhaustion():
    p, sleeps = make(fail_times=99, max_retries=2)
    assert p.process(ev(1)) == {"status": "dead_letter"}
    assert sleeps == [1, 2]
    assert len(p.dead_letter) == 1
    entry = p.dead_letter[0]
    assert entry["attempts"] == 3 and "transient" in entry["error"]


def test_unknown_type_dead_letters_without_retry():
    p, sleeps = make()
    assert p.process(ev(1, type_="mystery"))["status"] == "dead_letter"
    assert sleeps == []


def test_never_raises():
    p, _ = make(fail_times=99, max_retries=0)
    p.process(ev(1))


def test_replay_dead_letter():
    handler = flaky(4)
    sleeps = []
    p = EventProcessor({"job": handler}, max_retries=2, sleep=sleeps.append)
    p.process(ev(1))
    assert p.dead_letter
    recovered = p.replay_dead_letter()
    assert recovered == 1
    assert p.dead_letter == []
    assert "e1" in p.processed
