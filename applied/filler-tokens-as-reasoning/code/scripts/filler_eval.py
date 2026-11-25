"""
Entry point for the filler-token replication harness.

    python -m scripts.filler_eval --config runs/filler_baseline.json

The experiment node's fixed run command is `bash runs/filler_eval.sh`, which
calls this module with the committed config. To test a variant, branch a child
experiment and edit `runs/filler_baseline.json` on its branch; never change the
command.
"""
import sys

from nanochat.filler_eval import main

if __name__ == "__main__":
    sys.exit(main())
