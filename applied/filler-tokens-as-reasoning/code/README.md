# Not All Fillers Substitute for Reasoning

Replication and extension of the filler-token experiments from **Baherwani, Goldstein, & Panda (2026)**, *"Not All LLM Reasoning is Visible in the Chain-of-Thought"* (`arXiv:2607.22925`), evaluated on Qwen3 models served via OpenRouter.

Accompanying manuscript: `ste_filler_replication.tex` (ASI Research Lab, November 2026).

---

## Summary of Findings

- **Partial Replication (10-shot, $N=1{,}000$, Qwen3-235B-A22B-2507):** The arithmetic null replicates ($+0.57 \pm 1.63\text{ pp}$ vs. the paper's $-1.3\text{ pp}$), while the multiplication null does not ($+8.30 \pm 4.59\text{ pp}$ vs. $-0.4\text{ pp}$), alongside lower base rates on both tasks. Two 0-shot prompting diagnostics isolate the divergence to the FP8 OpenRouter serving path rather than prompt formatting.
- **Substitution Test (0-shot ASD-STE100 filler vs. Full CoT):** A $\sim 200$-token ASD-STE100 controlled-vocabulary filler lifts 4-digit multiplication from $34.20\%$ to $46.40\%$ ($+12.2\text{ pp}$), but still trails full chain-of-thought ($97.20\%$) by $51.5 \pm 7.1\text{ pp}$. On multi-step arithmetic, the filler regresses relative to baseline ($4.27\% \to 2.84\%$, $94.0 \pm 3.3\text{ pp}$ behind CoT at $96.50\%$).
- **Efficiency Inversion (KV-Cache & Decode Accounting):** Moving $\sim 200$ tokens from sequential decode into parallel prefill reduces decode-stage KV token-reads by $78\times$ (multiplication) and $170\times$ (arithmetic), peak KV by $0.42\times / 0.37\times$, and wall-clock time by $\sim 16\times$—at an accuracy cost of $-0.56$ to $-0.66\text{ pp}$ per $1\times$ KV-read reduction.
- **Recovery Grid ($2 \times 8 \times 2$ cells, $N=1{,}000$ each):**
  - **CoT accuracy is a cliff, not a curve:** $0\%$ at `max_tokens=128`, $0.9\text{--}3.5\%$ at `256`, $37.2\text{--}68.8\%$ at `512`, and $94.0\text{--}98.6\%$ at `2048`. The explicit `Answer:` cue rate confirms that sub-512 caps fail due to trace truncation rather than reasoning failure.
  - **Filler gains saturate and require high baseline capability:** STE gains on 235B peak at 200 tokens and turn negative by 500; on Qwen3-30B-A3B, filler gains never materialize at any length (cross-partial $-7.3 \pm 5.5\text{ pp}$ per 100 tokens per decade over 0–200 tokens).
  - **Pareto frontier:** The smaller 30B model with full CoT dominates ($98.6\%$ multiplication at $53.1\text{ MiB}$ peak KV vs. $97.2\%$ at $128.6\text{ MiB}$ for 235B).

---

## Repository Structure

```text
.
├── nanochat/                  # Standalone evaluation harness (stdlib + tiktoken)
│   ├── filler_eval.py         # Main evaluation loop, scoring, CIs, and reporting
│   ├── filler_tasks.py        # Deterministic task generators (multiplication, arithmetic, var counting)
│   ├── filler_tokens.py       # 18 filler streams (incl. counting & ASD-STE100) + token calibration
│   ├── filler_prompts.py      # Prefill, non-prefill, and full-CoT prompt builders
│   ├── filler_backends.py     # OpenRouter HTTP client, JSONL response cache, mock backend
│   └── orx_run.py             # Stdlib-only structured progress emitter
├── scripts/
│   └── filler_eval.py         # CLI entry point (`python -m scripts.filler_eval`)
├── runs/
│   ├── filler_eval.sh         # Fixed runner script (auto-manages tiktoken via uv)
│   ├── filler_baseline_10shot.json        # Sec. 4.1 / Table 1: 10-shot baseline vs. counting
│   ├── filler_baseline_0shot.json         # Sec. 4.1 Diagnostic 1: 0-shot multiplication
│   ├── filler_baseline_0shot_minimal.json # Sec. 4.1 Diagnostic 2: 0-shot minimal `/no_think` prompt
│   ├── filler_ste_cot.json                # Sec. 4.2–4.3 / Tables 2–3: ASD-STE100 vs. full CoT
│   └── filler_grid.json                   # Sec. 4.4 / Table 4: 2 models × 8 conditions × 2 tasks
├── tests/
│   └── test_filler_eval.py    # Hermetic offline unit & end-to-end tests
├── results/                   # Raw JSON evaluation outputs
├── figs/                      # Publication figures for the LaTeX manuscript
├── cache/                     # (Optional) Archived OpenRouter JSONL response caches
├── artifacts/                 # (Optional) Auxiliary analysis scripts and reports
├── ste_filler_replication.tex # Manuscript source
└── .python-version            # Python 3.10
```

---

## Requirements

- **Python 3.10+**
- **[`uv`](https://docs.astral.sh/uv/)** (recommended; `runs/filler_eval.sh` automatically runs in an isolated `uv` environment with `tiktoken==0.11.0`, requiring no PyTorch or GPU dependencies)
- **OpenRouter API key** for live model evaluations:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  ```

---

## Quick Start

### 1. Run the Offline Test Suite

Verifies task determinism, token calibration, prompt layouts, answer parsing, truncation tracking, wall-clock deadlines, and end-to-end scoring with no network calls:

```bash
uv run --no-project --with pytest --with "tiktoken==0.11.0" pytest tests/test_filler_eval.py -v
```

### 2. Preview Prompts (Dry Run)

Inspect the rendered prompts and calibrated filler lengths for any config without calling OpenRouter:

```bash
FILLER_EVAL_CONFIG=runs/filler_ste_cot.json bash runs/filler_eval.sh --dry-run
```

### 3. Replicate the Experiments

Run any experiment by setting `FILLER_EVAL_CONFIG`:

| Experiment | Paper Reference | Command |
|---|---|---|
| 10-shot replication (baseline vs. counting) | Sec. 4.1, Table 1 | `FILLER_EVAL_CONFIG=runs/filler_baseline_10shot.json bash runs/filler_eval.sh` |
| 0-shot multiplication diagnostic | Sec. 4.1 | `FILLER_EVAL_CONFIG=runs/filler_baseline_0shot.json bash runs/filler_eval.sh` |
| 0-shot minimal `/no_think` diagnostic | Sec. 4.1 | `FILLER_EVAL_CONFIG=runs/filler_baseline_0shot_minimal.json bash runs/filler_eval.sh` |
| ASD-STE100 filler vs. full CoT & KV accounting | Sec. 4.2–4.3, Tables 2–3 | `FILLER_EVAL_CONFIG=runs/filler_ste_cot.json bash runs/filler_eval.sh` |
| Recovery grid ($2 \times 8 \times 2$ sweep) | Sec. 4.4, Table 4 | `FILLER_EVAL_CONFIG=runs/filler_grid.json bash runs/filler_eval.sh` |

*(If you did not rename `runs/filler_baseline.json` to `runs/filler_baseline_0shot_minimal.json`, use `runs/filler_baseline.json` for Diagnostic 2.)*

### Useful CLI Flags

Extra arguments passed to `runs/filler_eval.sh` are forwarded directly to `scripts.filler_eval`:

- `--limit 50` — evaluate only the first `N` problems per cell (smoke test)
- `--models qwen/qwen3-30b-a3b-instruct-2507` — run a comma-separated subset of models
- `--conditions baseline,ste_filler,cot` — run a comma-separated subset of conditions (`baseline` must be included)
- `--no-report` — suppress `::orx::` structured progress lines on stdout

Example quick smoke test (20 problems, no structured progress lines):

```bash
FILLER_EVAL_CONFIG=runs/filler_ste_cot.json bash runs/filler_eval.sh --limit 20 --no-report
```

---

## Response Caching & Reproducibility

- **Deterministic tasks:** All tasks use fixed seeds (`12345` for multiplication, `54321` for arithmetic) with separate seeded draws for few-shot demonstrations, ensuring paired comparisons across conditions and models.
- **Disk cache:** Responses are cached in JSONL files keyed by the SHA-256 hash of the exact request body (including `model`, `messages`, `max_tokens`, and `reasoning`). By default, configs point to:
  - `~/.cache/nanochat/filler_eval_cache.jsonl`
  - `~/.cache/nanochat/filler_ste_cot_cache.jsonl`
- If you archived caches into `./cache/` and want to replay them on a fresh machine at zero API cost, copy them into `~/.cache/nanochat/` or update the `"cache"` path in the config JSONs. To force fresh live API queries, point `"cache"` to a new file or remove the `"cache"` key.
- **Outputs:** Each completed run writes timestamped results to `results/filler_eval_<YYYYMMDD-HHMMSS>.json` and updates `results/filler_eval_latest.json`.

---

## Building the Paper

To compile the manuscript PDF from `ste_filler_replication.tex` (requires a LaTeX distribution such as MacTeX/TeX Live and the PDF figures in `figs/`):

```bash
latexmk -pdf ste_filler_replication.tex
```

---

## Citation

If you build on this replication or extension, please cite the original work and this report:

```bibtex
@article{baherwani2026not,
  title   = {Not All LLM Reasoning is Visible in the Chain-of-Thought},
  author  = {Baherwani, Vatsal and Goldstein, Tom and Panda, Ashwinee},
  journal = {arXiv preprint arXiv:2607.22925},
  year    = {2026}
}
```