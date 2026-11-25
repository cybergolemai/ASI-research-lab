"""
Synthetic tasks from "Not All LLM Reasoning is Visible in the Chain-of-Thought"
(arXiv:2607.22925), Appendix A.2.

Three tasks, each a deterministic list of problems with an integer answer:

  - multiplication:     4-digit x 4-digit, seed 12345, 1,000 problems
  - arithmetic:         5-7 nested +, -, *, //, % operations, operands in [-99, 99]
  - variable_counting:  count the distinct variables assigned a value in a snippet

Every problem is generated from a fixed seed, so all models and all filler
conditions in one experiment see exactly the same problems. The paper's
comparisons are paired within a task (Section 3.2), and this is what makes that
possible.

Arithmetic answers use Python's integer semantics: `//` floors toward negative
infinity and `%` takes the sign of the divisor. The paper specifies the operator
set but not its rounding convention; problems whose value depends on that choice
are a small minority, and the gold value is always the one Python computes.
"""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Problem:
    """One evaluation problem: the user-visible prompt and its integer answer."""
    index: int
    prompt: str
    answer: int


# -----------------------------------------------------------------------------
# Multi-digit multiplication

MULTIPLICATION_SEED = 12345


def multiplication(n, seed=MULTIPLICATION_SEED):
    """4-digit x 4-digit multiplication, e.g. "What is 4821 times 9073?"."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        x = rng.randint(1000, 9999)
        y = rng.randint(1000, 9999)
        problems.append(Problem(i, f"What is {x} times {y}?", x * y))
    return problems


# -----------------------------------------------------------------------------
# Multi-step arithmetic

ARITHMETIC_SEED = 54321
ARITHMETIC_OPS = ("+", "-", "*", "//", "%")
ARITHMETIC_MAX_ABS = 10 ** 6
# A left-nested chain that ends in modulus leaves a one- or two-digit answer, and
# a short answer can be guessed without doing the arithmetic. The paper's own
# examples end in a multiplication and its 11% baseline implies answers that
# cannot be shortcut, so expressions below this magnitude are redrawn.
ARITHMETIC_MIN_ABS = 100


def _evaluate(node):
    if isinstance(node, int):
        return node
    op, left, right = node
    a, b = _evaluate(left), _evaluate(right)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "//":
        return a // b
    if op == "%":
        return a % b
    raise ValueError(f"unknown operator {op!r}")


def _format(node, top=True):
    if isinstance(node, int):
        return str(node)
    op, left, right = node
    body = f"{_format(left, False)} {op} {_format(right, False)}"
    return body if top else f"({body})"


def _build_expression(rng, n_ops):
    """Grow a left-nested expression by wrapping the running value in one op."""
    node = rng.randint(-99, 99)
    for _ in range(n_ops):
        op = rng.choice(ARITHMETIC_OPS)
        operand = rng.randint(-99, 99)
        if op in ("//", "%") and operand == 0:
            operand = 1  # never divide by zero
        # non-commutative ops always put the running value on the left
        if op in ("//", "%") or rng.random() < 0.5:
            node = (op, node, operand)
        else:
            node = (op, operand, node)
    return node


def arithmetic(n, seed=ARITHMETIC_SEED):
    """
    Problems with 5-7 nested operations over operands drawn from [-99, 99].

    Following Greenblatt (2025) as cited in Appendix A.2. Expressions are redrawn
    when |answer| falls below 100 or grows past 10^6: a short answer can be
    guessed without doing the arithmetic, and an enormous one cannot be emitted.
    """
    rng = random.Random(seed)
    problems = []
    while len(problems) < n:
        node = _build_expression(rng, rng.randint(5, 7))
        value = _evaluate(node)
        if not ARITHMETIC_MIN_ABS <= abs(value) <= ARITHMETIC_MAX_ABS:
            continue
        problems.append(Problem(len(problems), f"What is {_format(node)}?", value))
    return problems


# -----------------------------------------------------------------------------
# Variable counting in code

VARIABLE_SEED = 98765
_NAME_POOL = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "p",
    "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "total", "count", "result", "value", "data", "items", "tmp", "acc", "out", "buf",
]
_BINOPS = ("+", "*", "-")


def _code_prompt(rng):
    """A short function whose body assigns a known set of distinct variables."""
    n_distinct = rng.randint(3, 7)
    names = rng.sample(_NAME_POOL, n_distinct)
    lines = ["def f(x):"]
    assigned = []
    for name in names:
        if assigned and rng.random() < 0.6:
            rhs = f"{rng.choice(assigned)} {rng.choice(_BINOPS)} {rng.randint(1, 9)}"
        else:
            rhs = str(rng.randint(1, 99))
        lines.append(f"    {name} = {rhs}")
        assigned.append(name)
        # a reassignment does not add a distinct variable, so line count != answer
        if rng.random() < 0.3:
            lines.append(f"    {name} = {name} {rng.choice(_BINOPS)} {rng.randint(1, 9)}")
    lines.append(f"    return {rng.choice(assigned)}")
    code = "\n".join(lines)
    prompt = (
        "How many distinct variables are assigned a value in this code?\n\n"
        f"```python\n{code}\n```"
    )
    return prompt, n_distinct


def variable_counting(n, seed=VARIABLE_SEED):
    """Count distinct variables assigned a value (function parameters excluded)."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        prompt, answer = _code_prompt(rng)
        problems.append(Problem(i, prompt, answer))
    return problems


# -----------------------------------------------------------------------------

TASKS = {
    "multiplication": multiplication,
    "arithmetic": arithmetic,
    "variable_counting": variable_counting,
}


def build_task(name, n, seed=None):
    """Build `n` problems for a named task, deterministically."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASKS)}")
    if seed is None:
        return TASKS[name](n)
    return TASKS[name](n, seed=seed)
