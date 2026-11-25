# nanochat local pipeline results

This bundled demo records a completed local `runs/runcpu.sh` pipeline on Apple Silicon. It is historical evidence included with OpenResearch; onboarding does not rerun training on the new user's machine.

## Base training and evaluation

- Model: d6, 6 layers, 73.5M parameters, sequence length 512
- Training: 5,000 steps, 81.92M tokens
- Final training-loop validation BPB: 1.165758
- Base-eval train BPB: 1.152185
- Base-eval validation BPB: 1.119301
- CORE accuracy: wikidata 0.0000, openbook 0.2500, winogrande 0.5625, operators 0.0000

## Supervised fine-tuning

- Started from the validated base checkpoint with a fresh optimizer
- Training: 1,500 steps
- Validation BPB: 1.0174 → 1.0580 → 0.9914 → 0.9513 → 0.9141 → 0.8483 → 0.7950 → 0.7486 → 0.7389
- Final/minimum validation BPB: 0.7389

## Chat confirmation

The final command loaded the step-1499 SFT checkpoint on MPS and answered: “Paris … The capital of France is Paris.”

## Evidence pack

The `evidence/` directory contains the real checkpoint metadata, trained tokenizer, structured metrics, final inference transcript, and a manifest of both bundled and omitted run outputs. To reduce the bundled demo project's download size, large model weights, optimizer states, datasets, and environments are intentionally omitted.

## Portable setup repairs

- Reported each stage's progress, and training's loss, validation BPB and throughput, to OpenResearch.
- Shared downloaded shards, the trained tokenizer and uv's cache across runs; checkpoints stay in each run.
- Excluded conversations longer than the SFT row capacity and guarded all-masked batches.
- Disabled memory-heavy periodic ChatCORE while retaining validation BPB.
- Made the chat CLI reuse the project environment and recorded cache location.
