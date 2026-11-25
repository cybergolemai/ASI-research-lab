"""
Filler token sequences from "Not All LLM Reasoning is Visible in the
Chain-of-Thought" (arXiv:2607.22925), Table 6 and Appendix A.4.

A filler is a fixed token sequence that is semantically irrelevant to the
problem: the same words appear for every question, so nothing in the sequence
identifies a particular problem or answer.

Each sequence is defined as a stream of words. `build_filler` truncates that
stream so the resulting text is approximately `n_tokens` tokens under a chosen
tokenizer, which is the calibration the paper performs per model (Appendix A.4:
"we perform a binary search over word count, counting tokens using the model's
tokenizer"). The tokenizer here is whatever the experiment config names; it is
an approximation of the served model's tokenizer and is recorded in the output.
"""
import math
import random

FILLER_TYPES = (
    "counting",
    "fibonacci",
    "primes",
    "squares",
    "pi_digits",
    "random_numbers",
    "digit_words",
    "alphabet",
    "nato",
    "animals",
    "fruits",
    "colors",
    "states",
    "lorem",
    "pause",
    "ellipsis",
    "random_tokens",
    "ste_dictionary",
)

_DIGIT_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()

_NATO = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()

_ANIMALS = (
    "cat dog bear wolf lion tiger eagle shark horse mouse rabbit panda otter "
    "whale falcon turtle camel zebra koala moose lynx bison heron cobra gecko "
    "puma ferret badger salmon beetle robin"
).split()

_FRUITS = (
    "apple mango peach grape lemon banana cherry orange papaya melon kiwi plum "
    "guava lychee apricot coconut fig date pear lime olive berry currant "
    "pomelo quince tamarind durian"
).split()

_COLORS = (
    "red blue green yellow purple orange pink brown black white gray teal "
    "violet indigo maroon beige cyan magenta olive navy coral amber lilac "
    "crimson emerald ivory khaki lavender"
).split()

_STATES = (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "
    "Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky "
    "Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi "
    "Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania "
    "Tennessee Texas Utah Vermont Virginia Washington Wisconsin Wyoming"
).split()

_LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo"
).split()

# ASD-STE100 Simplified Technical English approved vocabulary. This is a
# representative subset of the controlled dictionary (the full standard lists
# roughly 900 approved words plus technical nouns). Unlike the paper's filler
# sequences, these are meaningful English words, so this type tests whether a
# controlled vocabulary used as filler preserves answer quality.
_STE_WORDS = (
    "a an the and or but if then when where which who what why how this that "
    "these those all any both each every some no not only also again always "
    "never often sometimes before after above below between during through "
    "about against into over under up down out off on in at by for with without "
    "adjust align apply assemble attach balance bleed check clean close connect "
    "cut decrease disconnect do examine fill find flush get give go have hold "
    "increase install keep let loosen lubricate make measure move open put "
    "remove repair replace rotate run set start stop tighten turn use wait "
    "add change compare complete correct count cut divide dry empty estimate "
    "extend fix follow heat inspect join lift limit lock mark mix obey oil "
    "operate paint position prepare press pull push read record release repeat "
    "rinse seal select separate shorten show slow supply support test tighten "
    "transfer try weigh work write "
    "air area bolt cable cap case connection cover damage end equipment filter "
    "fluid gauge hose indicator line liquid nut oil part pipe plug pressure "
    "pump screw seal surface system tank test tool valve wire bearing bracket "
    "clamp clip connector cylinder diaphragm disc drain fastener fitting flange "
    "gasket handle head housing inlet insert joint knob label lever lid lug "
    "motor mount nozzle outlet panel pin piston plate port pulley rod sample "
    "shaft sheet sleeve socket spacer spring switch terminal thread tube "
    "correct damaged dirty dry empty full good hot cold large small long short "
    "loose tight new old open closed safe clean clear direct easy hard heavy "
    "light rough smooth soft solid stable straight strong weak"
).split()

# 300 decimal digits of pi, split into single digits: 3 1 4 1 5 9 ...
_PI_DIGITS = (
    "314159265358979323846264338327950288419716939937510582097494459230781640"
    "628620899862803482534211706798214808651328230664709384460955058223172535"
    "940812848111745028410270193852110555964462294895493038196442881097566593"
    "344612847564823378678316527120190914564856692346034861045432664821339360"
)


def _cycle(words):
    """Yield `words` forever, restarting the list when it runs out."""
    i = 0
    while True:
        yield words[i % len(words)]
        i += 1


def _fibonacci():
    a, b = 1, 1
    while True:
        yield str(a)
        a, b = b, a + b


def _primes():
    candidate = 2
    while True:
        is_prime = True
        for d in range(2, int(candidate ** 0.5) + 1):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            yield str(candidate)
        candidate += 1


def _counting():
    i = 1
    while True:
        yield str(i)
        i += 1


def _squares():
    i = 1
    while True:
        yield str(i * i)
        i += 1


def _pi_digits():
    return _cycle(list(_PI_DIGITS))


def _random_numbers():
    # fixed seed: the paper keeps one random sequence across every problem
    rng = random.Random(0)
    while True:
        yield str(rng.randint(10, 99))


def _random_tokens():
    rng = random.Random(1)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    while True:
        yield "".join(rng.choice(alphabet) for _ in range(3))


def _stream(kind):
    if kind == "counting":
        return _counting()
    if kind == "fibonacci":
        return _fibonacci()
    if kind == "primes":
        return _primes()
    if kind == "squares":
        return _squares()
    if kind == "pi_digits":
        return _pi_digits()
    if kind == "random_numbers":
        return _random_numbers()
    if kind == "digit_words":
        return _cycle(_DIGIT_WORDS)
    if kind == "alphabet":
        return _cycle(list("abcdefghijklmnopqrstuvwxyz"))
    if kind == "nato":
        return _cycle(_NATO)
    if kind == "animals":
        return _cycle(_ANIMALS)
    if kind == "fruits":
        return _cycle(_FRUITS)
    if kind == "colors":
        return _cycle(_COLORS)
    if kind == "states":
        return _cycle(_STATES)
    if kind == "lorem":
        return _cycle(_LOREM)
    if kind == "pause":
        return _cycle(["pause"])
    if kind == "ellipsis":
        return _cycle(["..."])
    if kind == "random_tokens":
        return _random_tokens()
    if kind == "ste_dictionary":
        return _cycle(_STE_WORDS)
    raise ValueError(f"unknown filler type {kind!r}; expected one of {list(FILLER_TYPES)}")


def filler_text(kind, words):
    """Join the first `words` words of a filler stream with single spaces."""
    stream = _stream(kind)
    return " ".join(next(stream) for _ in range(words))


def count_tokens_whitespace(text):
    """Fallback counter used when tiktoken is unavailable: ~1.3 tokens per word."""
    return max(1, math.ceil(len(text.split()) * 1.3))


def get_token_counter(spec):
    """
    Resolve a tokenizer spec into a (count, resolved_name) pair.

    Supported specs:
      "tiktoken:<encoding>"  e.g. tiktoken:o200k_base, tiktoken:cl100k_base
      "whitespace"           word-count approximation, no dependency
    """
    if spec.startswith("tiktoken:"):
        encoding = spec.split(":", 1)[1]
        try:
            import tiktoken
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "tiktoken is required to calibrate filler length in tokens. "
                "Provision the project environment (`uv sync --extra cpu`) or set "
                "`\"tokenizer\": \"whitespace\"` in the config to accept a "
                "word-count approximation."
            ) from error
        enc = tiktoken.get_encoding(encoding)
        return (lambda text: len(enc.encode(text))), spec
    if spec == "whitespace":
        return count_tokens_whitespace, spec
    raise ValueError(f"unsupported tokenizer spec {spec!r}")


def build_filler(kind, n_tokens, counter, max_words=100_000, n_words=None):
    """
    Truncate a filler stream to approximately `n_tokens` tokens.

    Token count is non-decreasing in word count, so binary search finds the
    first word count that reaches the target; the neighbouring count is also
    scored and the closer of the two is kept (Appendix A.4's calibration).

    `n_words` bypasses calibration and takes exactly that many words. The paper's
    cross-model comparison uses counting 1-100, which is a fixed 100 words and
    tokenizes to roughly 200 tokens across models (Appendix A.4); that setting is
    specified by word count, not by token count.
    """
    if n_words is not None:
        text = filler_text(kind, int(n_words))
        return {
            "type": kind,
            "text": text,
            "words": int(n_words),
            "tokens": counter(text),
            "target_tokens": n_tokens,
            "calibration": "words",
        }
    if n_tokens <= 0:
        return {"type": kind, "text": "", "words": 0, "tokens": 0,
                "target_tokens": n_tokens, "calibration": "tokens"}

    lo, hi = 1, 1
    while counter(filler_text(kind, hi)) < n_tokens:
        lo = hi
        hi *= 2
        if hi > max_words:
            raise ValueError(f"{kind!r} cannot reach {n_tokens} tokens within {max_words} words")

    while lo < hi:
        mid = (lo + hi) // 2
        if counter(filler_text(kind, mid)) >= n_tokens:
            hi = mid
        else:
            lo = mid + 1

    candidates = sorted({max(1, lo - 1), lo, lo + 1})
    scored = [(abs(counter(filler_text(kind, w)) - n_tokens), w) for w in candidates]
    _, words = min(scored)
    text = filler_text(kind, words)
    return {
        "type": kind,
        "text": text,
        "words": words,
        "tokens": counter(text),
        "target_tokens": n_tokens,
        "calibration": "tokens",
    }
