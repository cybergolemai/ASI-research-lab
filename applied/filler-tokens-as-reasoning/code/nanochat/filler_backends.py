"""
Model backends for the filler-token replication harness (arXiv:2607.22925).

Three backends:

  openrouter  the paper's own serving path: chat completions via OpenRouter,
              with assistant prefill and reasoning disabled (Section 3.2, A.3).
  local       a nanochat checkpoint served in-process. Requires the run
              workspace (`bash runs/runcpu.sh`) to have produced weights.
  mock        a deterministic stand-in used by tests and --dry-run. It never
              produces a scientific result.

The harness talks to OpenRouter with the standard library only, so the run
command needs no extra dependency beyond the project environment.
"""
from dataclasses import dataclass
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


class BackendError(RuntimeError):
    """A request failed in a way the caller may need to handle (e.g. no prefill)."""

    def __init__(self, message, status=None, body=""):
        super().__init__(message)
        self.status = status
        self.body = body


# -----------------------------------------------------------------------------
# Response cache
#
# Reruns and sibling experiments ask many identical questions, so responses are
# cached on disk and keyed by the full request. This makes a repaired or resumed
# run cheap without changing what is measured.

class ResponseCache:
    def __init__(self, path):
        self.path = os.path.expanduser(path) if path else None
        self._lock = threading.Lock()
        self._entries = {}
        if self.path and os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self._entries[entry["key"]] = entry
                    except (json.JSONDecodeError, KeyError):
                        continue

    @staticmethod
    def key(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        if not self.path:
            return
        with self._lock:
            self._entries[key] = value
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(value) + "\n")


# -----------------------------------------------------------------------------
# OpenRouter

class OpenRouterBackend:
    """Chat completions through OpenRouter, matching the paper's serving path."""

    name = "openrouter"

    def __init__(self, model, api_key=None, temperature=1.0, top_p=0.95, max_tokens=32,
                 timeout=180, retries=4, reasoning=False, cache=None, base_url=OPENROUTER_URL,
                 deadline=240):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise BackendError(
                "OPENROUTER_API_KEY is not set. Add it to the project environment "
                "before launching an openrouter-backend run."
            )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.deadline = deadline
        self.retries = retries
        self.reasoning = reasoning
        self.cache = cache
        self.base_url = base_url

    def _payload(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        # The paper disables reasoning for API models through OpenRouter.
        if not self.reasoning:
            body["reasoning"] = {"enabled": False}
        return body

    def _post(self, body):
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "filler-token-replication",
            },
            method="POST",
        )

        def fetch():
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        # urllib's timeout bounds each blocking socket operation, not the whole
        # request: a response that trickles in keeps resetting it, so one stuck
        # read can hold the caller far past `timeout`. The calls are already on
        # worker threads and the caller consumes futures in submission order, so
        # a single hung request silences every later completion and the run looks
        # stalled. Enforce a wall-clock deadline here and abandon the thread.
        outcome = {}

        def target():
            try:
                outcome["value"] = fetch()
            except BaseException as error:  # surfaced in the calling thread
                outcome["error"] = error

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self.deadline)
        if "value" in outcome:
            return outcome["value"]
        if "error" in outcome:
            raise outcome["error"]
        raise TimeoutError(
            f"no complete response within {self.deadline}s (socket timeout {self.timeout}s)"
        )

    def complete(self, messages):
        body = self._payload(messages)
        key = ResponseCache.key(body)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return Completion(hit["text"], hit.get("prompt_tokens", 0),
                                  hit.get("completion_tokens", 0), cached=True)

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                payload = self._post(body)
                choice = payload["choices"][0]
                text = choice["message"].get("content") or ""
                usage = payload.get("usage") or {}
                result = Completion(
                    text,
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
                if self.cache is not None:
                    self.cache.put(key, {
                        "key": key,
                        "text": result.text,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    })
                return result
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:2000]
                # 4xx other than rate limiting is a request problem, not a transient one
                if error.code == 429 or error.code >= 500:
                    last_error = BackendError(f"HTTP {error.code}", error.code, detail)
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise BackendError(f"HTTP {error.code}", error.code, detail) from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = BackendError(f"request failed: {error}")
                time.sleep(min(2 ** attempt, 30))
        raise last_error or BackendError("request failed")


# -----------------------------------------------------------------------------
# Mock

class MockBackend:
    """
    Deterministic fake model for tests, smoke runs, and --dry-run.

    It solves multiplication and arithmetic problems itself with a probability
    that depends on whether a filler sequence is present, so the plumbing
    (prompting, scoring, aggregation, deltas) can be exercised offline. Results
    from this backend are marked non-scientific by the caller.
    """

    def __init__(self, model="mock/model", filler_uplift=0.15, seed=0):
        self.model = model
        self.filler_uplift = filler_uplift
        self.seed = seed
        self.name = "mock"

    @staticmethod
    def _solve(prompt):
        match = re.search(r"What is (\d+) times (\d+)\?", prompt)
        if match:
            return int(match.group(1)) * int(match.group(2))
        match = re.search(r"What is (.+)\?$", prompt, flags=re.S)
        if match:
            expression = match.group(1)
            if re.fullmatch(r"[\d\s\+\-\*\/%\(\)]+", expression):
                try:
                    return eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - generated input only
                except Exception:
                    return None
        return None

    def complete(self, messages):
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        # with a filler, the final assistant turn starts with the filler sequence
        last_assistant = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
        has_filler = bool(last_assistant and not last_assistant["content"].startswith("Answer:"))
        answer = self._solve(question)
        if answer is None:
            return Completion(" ?")
        rng = random.Random(f"{self.seed}:{hashlib.sha256(question.encode()).hexdigest()}")
        correct = rng.random() < (0.35 + (self.filler_uplift if has_filler else 0.0))
        return Completion(f" {answer}" if correct else f" {answer + rng.randint(1, 97)}")


# -----------------------------------------------------------------------------
# Local nanochat checkpoint

class LocalNanochatBackend:
    """
    Serve a nanochat SFT checkpoint in-process.

    The checkpoint is loaded lazily so that importing this module never pulls in
    torch. Weights are not part of the demo evidence pack: run `bash runs/runcpu.sh`
    first to produce `chatsft_checkpoints/` under NANOCHAT_BASE_DIR. A 6-layer
    model has no realistic chance at 4-digit multiplication, so treat this backend
    as a sanity check of the harness, not a replication of the paper.
    """

    name = "local"

    def __init__(self, model="nanochat-sft", source="sft", model_tag=None, device=None,
                 temperature=1.0, top_k=50, max_tokens=32, seed=42):
        self.model = model
        self.source = source
        self.model_tag = model_tag
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.seed = seed
        self._engine = None
        self._tokenizer = None

    def _load(self):
        if self._engine is not None:
            return
        from nanochat.checkpoint_manager import load_model
        from nanochat.common import autodetect_device_type
        from nanochat.engine import Engine

        device_type = self.device or autodetect_device_type()
        model, tokenizer, _ = load_model(self.source, device_type, self.model_tag)
        model.eval()
        self._engine = Engine(model, tokenizer)
        self._tokenizer = tokenizer

    def complete(self, messages):
        self._load()
        tokenizer = self._tokenizer
        # Render the conversation up to the final user turn, then start the
        # assistant turn ourselves so a prefill can be continued.
        prefill = ""
        if messages[-1]["role"] == "assistant":
            prefill = messages[-1]["content"]
            messages = messages[:-1]
        tokens, _ = tokenizer.render_conversation({"messages": messages})
        tokens = tokens + [tokenizer.encode_special("<|assistant_start|>")] + tokenizer.encode(prefill)

        generated = []
        for token_column, _ in self._engine.generate(
            tokens, num_samples=1, max_tokens=self.max_tokens,
            temperature=self.temperature, top_k=self.top_k, seed=self.seed,
        ):
            token = token_column[0]
            if token == tokenizer.encode_special("<|assistant_end|>"):
                break
            generated.append(token)
        return Completion(tokenizer.decode(generated))


# -----------------------------------------------------------------------------

def make_backend(spec, sampling, cache):
    """Build a backend from a config's backend name and sampling settings."""
    name = spec.get("backend", "openrouter")
    common = {
        "temperature": sampling.get("temperature", 1.0),
        "top_p": sampling.get("top_p", 0.95),
        "max_tokens": sampling.get("max_tokens", 32),
    }
    if name == "openrouter":
        # A request's whole retry budget must fit inside the run's stall window:
        # the watcher stops a run that reports nothing for 10 minutes, and a hung
        # connection costs one timeout per attempt. The defaults (180s x 5) can
        # therefore go silent for ~15 minutes on a single stuck request. A config
        # may tighten them through `request`, which is not part of the cache key.
        request = spec.get("request") or {}
        return OpenRouterBackend(
            spec["model"],
            cache=cache,
            reasoning=sampling.get("reasoning", False),
            timeout=request.get("timeout", 180),
            deadline=request.get("deadline", 240),
            retries=request.get("retries", 4),
            **common,
        )
    if name == "mock":
        return MockBackend(model=spec["model"], seed=sampling.get("seed", 0))
    if name == "local":
        return LocalNanochatBackend(
            model=spec["model"],
            source=spec.get("source", "sft"),
            model_tag=spec.get("model_tag"),
            temperature=common["temperature"],
            max_tokens=common["max_tokens"],
        )
    raise ValueError(f"unknown backend {name!r}; expected openrouter, local, or mock")
