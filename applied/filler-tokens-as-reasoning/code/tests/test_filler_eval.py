"""
Tests for the filler-token replication harness (arXiv:2607.22925).

These are hermetic: no network, no model weights. The end-to-end test drives the
mock backend so the whole pipeline (prompting, scoring, aggregation, deltas)
runs offline.

    python -m pytest tests/test_filler_eval.py -v
"""
import pytest

from nanochat.filler_backends import MockBackend, ResponseCache
from nanochat.filler_prompts import COT_SYSTEM_PROMPT, SYSTEM_PROMPT, build_messages
from nanochat.filler_tasks import arithmetic, build_task, multiplication, variable_counting
from nanochat.filler_tokens import FILLER_TYPES, build_filler, count_tokens_whitespace, filler_text
from nanochat.filler_eval import (
    Tally,
    build_conditions,
    confidence_interval,
    evaluate,
    looks_like_reasoning,
    parse_answer,
    parse_answer_cot,
)


# -----------------------------------------------------------------------------
# Tasks

def test_multiplication_is_deterministic_and_correct():
    problems = multiplication(5)
    assert problems == multiplication(5)
    assert len(problems) == 5
    for problem in problems:
        x, y = problem.prompt.removeprefix("What is ").removesuffix("?").split(" times ")
        assert len(x) == 4 and len(y) == 4
        assert problem.answer == int(x) * int(y)


def test_arithmetic_matches_python_semantics():
    problems = arithmetic(20)
    assert problems == arithmetic(20)
    for problem in problems:
        expression = problem.prompt.removeprefix("What is ").removesuffix("?")
        assert problem.answer == eval(expression, {"__builtins__": {}}, {})
        assert 100 <= abs(problem.answer) <= 10 ** 6


def test_variable_counting_counts_distinct_names():
    for problem in variable_counting(20):
        body = problem.prompt.split("```python\n")[1].rsplit("\n```", 1)[0]
        assigned = set()
        for line in body.splitlines()[1:]:
            stripped = line.strip()
            if " = " in stripped and not stripped.startswith("return"):
                assigned.add(stripped.split(" = ")[0])
        assert problem.answer == len(assigned)


def test_build_task_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_task("nope", 1)


# -----------------------------------------------------------------------------
# Fillers

def test_every_filler_type_produces_text_and_is_deterministic():
    for kind in FILLER_TYPES:
        text = filler_text(kind, 5)
        assert text == filler_text(kind, 5)
        assert text.strip()
    assert set(FILLER_TYPES) == {
        "counting", "fibonacci", "primes", "squares", "pi_digits", "random_numbers",
        "digit_words", "alphabet", "nato", "animals", "fruits", "colors", "states",
        "lorem", "pause", "ellipsis", "random_tokens", "ste_dictionary",
    }


def test_calibration_lands_near_target():
    # a counter that maps one word to one token, so the target is exact
    word_counter = lambda text: len(text.split())
    filler = build_filler("counting", 100, word_counter)
    assert filler["target_tokens"] == 100
    assert filler["tokens"] == 100
    # counting 1..100 is one word per number
    assert filler["words"] == 100


def test_calibration_handles_zero():
    filler = build_filler("pause", 0, count_tokens_whitespace)
    assert filler["text"] == ""
    assert filler["tokens"] == 0


# -----------------------------------------------------------------------------
# Prompting

def test_prompt_layout_with_prefill():
    problems = multiplication(2)
    examples = multiplication(2)
    filler = build_filler("counting", 100, count_tokens_whitespace)
    messages = build_messages(problems[0], examples, filler["text"], prefill=True)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    # 1 system + 2 * (user, assistant) + user + prefilled assistant
    assert len(messages) == 7
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"].endswith("Answer:")
    assert filler["text"] in messages[-1]["content"]
    # the filler never leaks into a user turn
    assert all(filler["text"] not in m["content"] for m in messages if m["role"] == "user")


def test_prompt_without_filler_has_no_filler_text():
    messages = build_messages(multiplication(1)[0], multiplication(1), "", prefill=True)
    assert messages[-1]["content"] == "Answer:"


def test_prompt_without_prefill_puts_cue_in_user_turn():
    messages = build_messages(multiplication(1)[0], multiplication(1), "1 2 3", prefill=False)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("Answer:")
    assert "1 2 3" in messages[-1]["content"]


def test_cot_prompt_has_no_prefill_and_uses_cot_instruction():
    messages = build_messages(multiplication(1)[0], [], "", cot=True)
    assert messages[0] == {"role": "system", "content": COT_SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == multiplication(1)[0].prompt
    # no assistant prefill and no filler in the CoT layout
    assert all(m["role"] != "assistant" for m in messages)


# -----------------------------------------------------------------------------
# Scoring

@pytest.mark.parametrize("text,expected", [
    (" 1234", 1234),
    ("Answer: 1234", 1234),
    ("Answer: -17", -17),
    ("1,234", 1234),
    ("N/A", None),
    ("", None),
    (" ?", None),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize("text,expected", [
    (" 1234", False),
    ("Answer: 1234", False),
    ("Let me think about this. The product is 1234", True),
    ("1 2 3 4 5 6 7 8 9 10 11 12 13 14", True),
])
def test_reasoning_detection(text, expected):
    assert looks_like_reasoning(text) is expected


def test_tally_discards_reasoning_samples():
    tally = Tally()
    tally.add(" 10", 10)                     # correct, kept
    tally.add(" 11", 10)                     # wrong, kept
    tally.add("Let me compute: 10", 10)      # correct but reasoning -> discarded
    summary = tally.summary()
    assert summary["n"] == 3
    assert summary["n_kept"] == 2
    assert summary["accuracy"] == 50.0
    assert summary["accuracy_including_discarded"] == pytest.approx(66.67, abs=0.01)
    assert summary["reasoning_rate"] == pytest.approx(33.33, abs=0.01)


@pytest.mark.parametrize("text,expected", [
    ("Let me think. 1234 * 5678 = 7006652. Answer: 7006652", 7006652),
    ("Answer: -17", -17),
    ("I estimate 1234, but the answer is 1235", 1235),
    ("", None),
    ("N/A", None),
])
def test_parse_answer_cot(text, expected):
    assert parse_answer_cot(text) == expected


def test_tally_keeps_reasoning_in_cot_mode():
    tally = Tally()
    tally.add("Let me compute. Answer: 10", 10, cot=True)   # kept, correct
    tally.add("I think 11. Answer: 11", 10, cot=True)       # kept, wrong
    summary = tally.summary()
    assert summary["n"] == 2
    assert summary["n_kept"] == 2
    assert summary["n_reasoning_discarded"] == 0
    assert summary["accuracy"] == 50.0


def test_confidence_interval_shrinks_with_n():
    assert confidence_interval(0.5, 100) > confidence_interval(0.5, 1000)
    assert confidence_interval(0.5, 0) == 0.0


# -----------------------------------------------------------------------------
# End to end

def test_mock_run_produces_paired_deltas(tmp_path):
    config = {
        "schema_version": 1,
        "backend": "mock",
        "models": ["mock/model"],
        "tasks": ["multiplication"],
        "conditions": [
            {"name": "baseline", "filler_type": None, "filler_tokens": 0},
            {"name": "counting_100", "filler_type": "counting", "filler_tokens": 100},
        ],
        "shots": 2,
        "n_problems": 40,
        "seed": 12345,
        "tokenizer": "whitespace",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32, "seed": 0},
        "prompting": {"prefill": True},
        "concurrency": 4,
        "max_samples": 5,
        "cache": str(tmp_path / "cache.jsonl"),
    }
    results = evaluate(config, report=False)
    cells = {(c["condition"]): c for c in results["cells"]}
    assert set(cells) == {"baseline", "counting_100"}
    assert cells["baseline"]["n"] == 40
    assert "delta_vs_baseline" not in cells["baseline"]
    assert "delta_vs_baseline" in cells["counting_100"]
    assert len(cells["baseline"]["samples"]) == 5
    assert any("mock backend" in w for w in results["warnings"])
    # the same config yields the same paired result
    again = evaluate(config, report=False)
    assert [c["accuracy"] for c in again["cells"]] == [c["accuracy"] for c in results["cells"]]


def test_response_cache_roundtrip(tmp_path):
    cache = ResponseCache(str(tmp_path / "c.jsonl"))
    key = ResponseCache.key({"a": 1})
    assert cache.get(key) is None
    cache.put(key, {"key": key, "text": " 7"})
    assert ResponseCache(str(tmp_path / "c.jsonl")).get(key)["text"] == " 7"


def test_conditions_calibrate_filler():
    config = {
        "conditions": [
            {"name": "baseline", "filler_type": None},
            {"name": "counting_100", "filler_type": "counting", "filler_tokens": 100},
        ]
    }
    conditions = build_conditions(config, count_tokens_whitespace)
    assert conditions[0]["filler"]["text"] == ""
    assert abs(conditions[1]["filler"]["tokens"] - 100) <= 5


# -----------------------------------------------------------------------------
# OpenRouter request contract (no network)

def test_openrouter_payload_and_response_parsing(monkeypatch):
    from nanochat.filler_backends import OpenRouterBackend

    backend = OpenRouterBackend("qwen/qwen3-235b-a22b-2507", api_key="test",
                                temperature=1.0, top_p=0.95, max_tokens=32)
    captured = {}

    def fake_post(body):
        captured.update(body)
        return {
            "choices": [{"message": {"content": " 1234"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 3},
        }

    monkeypatch.setattr(backend, "_post", fake_post)
    result = backend.complete([{"role": "user", "content": "hi"}])

    assert result.text == " 1234"
    assert result.prompt_tokens == 900
    assert captured["model"] == "qwen/qwen3-235b-a22b-2507"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    # the paper disables reasoning for API models through OpenRouter
    assert captured["reasoning"] == {"enabled": False}


def test_openrouter_uses_cache_on_repeat(monkeypatch, tmp_path):
    from nanochat.filler_backends import OpenRouterBackend

    backend = OpenRouterBackend("qwen/x", api_key="test",
                                cache=ResponseCache(str(tmp_path / "c.jsonl")))
    calls = []

    def fake_post(body):
        calls.append(body)
        return {"choices": [{"message": {"content": " 7"}}], "usage": {}}

    monkeypatch.setattr(backend, "_post", fake_post)
    messages = [{"role": "user", "content": "hi"}]
    first = backend.complete(messages)
    second = backend.complete(messages)

    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert second.text == " 7"


# -----------------------------------------------------------------------------
# Per-condition token caps (multi-CoT grids)

def test_conditions_carry_per_condition_max_tokens():
    config = {
        "conditions": [
            {"name": "baseline", "filler_type": None},
            {"name": "ste_100", "filler_type": "counting", "filler_tokens": 100},
            {"name": "cot_128", "cot": True, "max_tokens": 128},
            {"name": "cot", "cot": True},
        ]
    }
    conditions = {c["name"]: c for c in build_conditions(config, count_tokens_whitespace)}
    assert conditions["baseline"]["max_tokens"] is None
    assert conditions["ste_100"]["max_tokens"] is None
    assert conditions["cot_128"]["max_tokens"] == 128
    assert conditions["cot"]["max_tokens"] == 2048
    assert conditions["cot_128"]["reasoning"] is True
    assert conditions["cot"]["reasoning"] is True


def test_cot_caps_get_separate_backends_and_one_reference(tmp_path, monkeypatch):
    import nanochat.filler_eval as fe

    asked = []
    real_make_backend = fe.make_backend

    def recording_make_backend(spec, sampling, cache):
        asked.append({**sampling})
        return real_make_backend({**spec, "backend": "mock"}, sampling, cache)

    monkeypatch.setattr(fe, "make_backend", recording_make_backend)

    config = {
        "schema_version": 1,
        "backend": "mock",
        "models": ["mock/model"],
        "tasks": ["multiplication"],
        "conditions": [
            {"name": "baseline", "filler_type": None, "filler_tokens": 0},
            {"name": "ste_100", "filler_type": "counting", "filler_tokens": 100},
            {"name": "cot_128", "cot": True, "max_tokens": 128},
            {"name": "cot", "cot": True, "max_tokens": 2048},
        ],
        "reference_condition": "cot",
        "primary_condition": "ste_100",
        "shots": 0,
        "n_problems": 8,
        "seed": 12345,
        "tokenizer": "whitespace",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32},
        "prompting": {"prefill": True},
        "concurrency": 2,
        "max_samples": 2,
        "cache": str(tmp_path / "cache.jsonl"),
    }

    results = fe.evaluate(config, report=False)

    # one backend per distinct (max_tokens, reasoning): 32, 128 and 2048
    assert sorted({s["max_tokens"] for s in asked}) == [32, 128, 2048]
    assert len(asked) == 3, "baseline and ste_100 share the shared-sampling backend"

    cells = {c["condition"]: c for c in results["cells"]}
    assert cells["baseline"]["max_tokens"] == 32
    assert cells["cot_128"]["max_tokens"] == 128
    assert cells["cot"]["max_tokens"] == 2048
    assert cells["cot_128"]["reasoning"] is True

    # every cell names its reference, and the delta is measured against full CoT
    assert all(c["reference_condition"] == "cot" for c in results["cells"])
    expected = round(cells["cot_128"]["accuracy"] - cells["cot"]["accuracy"], 2)
    assert cells["cot_128"]["delta_vs_cot"] == expected
    assert cells["cot_128"]["delta_vs_reference"] == expected


def test_cache_key_distinguishes_token_caps():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 128}
    longer = {**body, "max_tokens": 2048}
    assert ResponseCache.key(body) != ResponseCache.key(longer)


def test_request_budget_comes_from_config_and_not_the_cache_key(monkeypatch):
    """A hung request must not outlast the run's stall window.

    The watcher stops a run silent for 10 minutes, so the retry budget has to be
    configurable and has to fit inside it. Timeout and retries are not part of
    the request body, so tightening them must not invalidate cached responses.
    """
    import nanochat.filler_backends as fb

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    spec = {
        "backend": "openrouter",
        "model": "qwen/qwen3-235b-a22b-2507",
        "request": {"timeout": 90, "retries": 3},
    }
    backend = fb.make_backend(spec, {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32}, None)
    assert backend.timeout == 90
    assert backend.retries == 3

    default = fb.make_backend({"backend": "openrouter", "model": "m"}, {}, None)
    assert (default.timeout, default.retries) == (180, 4)

    # worst case: 4 attempts at 90s plus 2+4+8s of backoff, under the 10m rule
    worst_case = 4 * 90 + 14
    assert worst_case < 10 * 60

    body = backend._payload([{"role": "user", "content": "hi"}])
    assert "timeout" not in body and "retries" not in body
    assert "deadline" not in body


def test_hung_response_is_cut_off_by_the_wall_clock_deadline(monkeypatch):
    """A trickling response must not hold a worker past `deadline`.

    urllib's timeout applies per socket operation, so a response that keeps
    arriving slowly never trips it. The caller consumes futures in submission
    order, so one such request silences the whole run.
    """
    import time as _time
    import urllib.request as _req

    from nanochat.filler_backends import OpenRouterBackend

    backend = OpenRouterBackend("m", api_key="k", timeout=90, deadline=0.2, retries=0)

    class SlowResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            _time.sleep(5)          # far past the deadline
            return b"{}"

    monkeypatch.setattr(_req, "urlopen", lambda *a, **k: SlowResponse())
    started = _time.monotonic()
    with pytest.raises(TimeoutError):
        backend._post({"x": 1})
    assert _time.monotonic() - started < 2.0, "the deadline, not the read, must bound the call"


def test_deadline_is_retried_and_then_surfaces(monkeypatch):
    """A deadline hit is transient: it retries, then reports the failure."""
    from nanochat.filler_backends import OpenRouterBackend

    backend = OpenRouterBackend("m", api_key="k", deadline=0.05, retries=1)
    calls = []

    def always_hang(body):
        calls.append(body)
        raise TimeoutError("no complete response")

    monkeypatch.setattr(backend, "_post", always_hang)
    with pytest.raises(Exception):
        backend.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 2, "one attempt plus one retry"


def test_tally_counts_answered_and_truncated_cot():
    tally = Tally()
    # a complete CoT trace, then two traces cut off before "Answer:"
    tally.add("so 12*12=144.\nAnswer: 144", 144, cot=True)
    tally.add("first 12*12 is 144, then I need to check", 144, cot=True)
    tally.add("", 7, cot=True)
    summary = tally.summary()
    assert summary["n"] == 3
    # the parser falls back to the last number, so only the empty trace is
    # unanswered; only the cued trace is an explicit answer
    assert summary["n_answered"] == 2
    assert summary["answer_rate"] == pytest.approx(66.67, abs=0.01)
    assert summary["n_explicit"] == 1
    assert summary["explicit_rate"] == pytest.approx(33.33, abs=0.01)
    assert summary["accuracy"] == pytest.approx(66.67, abs=0.01)


def test_reporting_opens_a_named_one_step_setup_stage(capsys):
    import json

    config = {
        "schema_version": 1,
        "backend": "mock",
        "models": ["mock/model"],
        "tasks": ["multiplication"],
        "conditions": [{"name": "baseline", "filler_type": None, "filler_tokens": 0}],
        "shots": 0,
        "n_problems": 4,
        "seed": 12345,
        "tokenizer": "whitespace",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32},
        "prompting": {"prefill": True},
        "concurrency": 2,
        "max_samples": 1,
    }
    evaluate(config, report=True)
    out = capsys.readouterr().out
    records = [json.loads(line[len("::orx:: "):]) for line in out.splitlines()
               if line.startswith("::orx:: ")]
    phased = [r for r in records if "phase" in r]
    # an opening record with no phase or total is what orx reports as an
    # instrumentation gap; the first stage must be named and bounded
    assert phased[0]["phase"] == "setup"
    assert phased[0]["of"] == 1
    assert phased[0]["step"] == 0
