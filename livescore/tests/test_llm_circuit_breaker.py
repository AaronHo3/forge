"""Phase 2: circuit breaker around the Claude director.

Without it, a persistently failing API (bad key, outage) was retried every
DIRECTOR_INTERVAL forever, spamming logs. The breaker trips after a few
consecutive failures and pauses calls for a cooldown; the music keeps playing
on the last style. These test the pure breaker state machine, no network.
"""

import pytest

from llm_style_director import LLMStyleDirector


def make_director() -> LLMStyleDirector:
    # __init__ does no I/O and never touches analyzer/controller, so None is fine.
    return LLMStyleDirector(analyzer=None, controller=None)


class _FakeMessages:
    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return type("Resp", (), {"content": [type("C", (), {"text": self._text})()]})()


class _FakeClient:
    def __init__(self, text: str):
        self.messages = _FakeMessages(text)


class _FakeController:
    def __init__(self):
        self.prompts = None

    def set_prompts(self, a, b, key=None):
        self.prompts = (a, b, key)


@pytest.mark.unit
class TestCircuitBreaker:
    def test_starts_closed(self):
        assert make_director()._circuit_open() is False

    def test_opens_only_at_threshold(self):
        d = make_director()
        for _ in range(d.CIRCUIT_THRESHOLD - 1):
            d._record_call_failure()
            assert d._circuit_open() is False   # not tripped yet
        d._record_call_failure()                # threshold reached
        assert d._circuit_open() is True

    def test_success_resets_failure_count(self):
        d = make_director()
        for _ in range(d.CIRCUIT_THRESHOLD - 1):
            d._record_call_failure()
        d._record_call_success()
        # after a success it takes the full threshold again to trip
        for _ in range(d.CIRCUIT_THRESHOLD - 1):
            d._record_call_failure()
        assert d._circuit_open() is False

    def test_closes_after_cooldown_elapses(self):
        d = make_director()
        for _ in range(d.CIRCUIT_THRESHOLD):
            d._record_call_failure()
        assert d._circuit_open() is True
        d._circuit_open_until = 0.0   # simulate the cooldown elapsing
        assert d._circuit_open() is False

    def test_success_clears_open_circuit(self):
        d = make_director()
        for _ in range(d.CIRCUIT_THRESHOLD):
            d._record_call_failure()
        assert d._circuit_open() is True
        d._record_call_success()
        assert d._circuit_open() is False

    def test_fresh_grace_period_after_cooldown(self):
        d = make_director()
        for _ in range(d.CIRCUIT_THRESHOLD):
            d._record_call_failure()
        d._circuit_open_until = 0.0          # simulate the cooldown elapsing
        assert d._circuit_open() is False
        d._record_call_failure()             # one failure must NOT instantly re-trip
        assert d._circuit_open() is False


@pytest.mark.unit
class TestBreakerGatesCalls:
    def test_open_circuit_skips_a_normal_call(self):
        d = make_director()
        d._controller = _FakeController()
        d._client = _FakeClient("unused")
        for _ in range(d.CIRCUIT_THRESHOLD):
            d._record_call_failure()
        d._process_text("she walked quietly", force=False)
        assert d._client.messages.calls == 0     # skipped while the breaker is open
        assert d._controller.prompts is None

    def test_force_inject_bypasses_open_circuit(self):
        d = make_director()
        d._controller = _FakeController()
        d._client = _FakeClient('Warm Felt Piano", "b": "Soft Rhodes", "key": "C major"}')
        for _ in range(d.CIRCUIT_THRESHOLD):
            d._record_call_failure()
        assert d._circuit_open() is True
        d._process_text("the hero charged in", force=True)   # operator intent wins
        assert d._client.messages.calls == 1
        assert d._controller.prompts == ("Warm Felt Piano", "Soft Rhodes", "C major")
