#!/bin/bash

# Baseline filler-token replication for arXiv:2607.22925
# ("Not All LLM Reasoning is Visible in the Chain-of-Thought").
#
# Run as:
#   bash runs/filler_eval.sh
#
# The command is fixed across the experiment tree. Every setting that varies
# between nodes lives in the committed config, not in this script:
#   runs/filler_baseline.json
#
# Requires OPENROUTER_API_KEY in the environment for the openrouter backend.
# Responses are cached under ~/.cache/nanochat/ so a repaired or resumed run
# does not pay for the same question twice.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${FILLER_EVAL_CONFIG:-runs/filler_baseline.json}"
echo "config: $CONFIG"

# The harness itself is standard library only; tiktoken calibrates filler length
# in tokens, as the paper does per model (Appendix A.4). Prefer a provisioned
# project environment, then an isolated uv environment, then the system Python.
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import tiktoken" >/dev/null 2>&1; then
    PYTHON=".venv/bin/python"
    echo "interpreter: $($PYTHON -c 'import sys; print(sys.executable, sys.version.split()[0])')"
    exec "$PYTHON" -m scripts.filler_eval --config "$CONFIG" "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing it the same way runs/runcpu.sh does"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "interpreter: isolated uv environment with tiktoken"
exec uv run --no-project --with "tiktoken==0.11.0" \
    python -m scripts.filler_eval --config "$CONFIG" "$@"
