# Fine-tuning the back-office model family

Two-track training plan (design spec section 13.2):

1. **ReconForge 1.7B continuity** (the published result): Qwen3-1.7B LoRA,
   ~2h on an Apple M5, MLX. This is the laptop/edge baseline and the
   reproducibility anchor.
2. **Qwen3.8-27B adapter candidate**: the flagship open-weight model (Apache
   2.0, office-workflows positioning, 262K native context). Trains on Kaggle
   free GPUs; verify Unsloth/MLX LoRA support for its hybrid
   DeltaNet+attention architecture before committing compute.

## Data

```sh
uv run python -m experiments.finetune.export_training_data \
    --out data/train.jsonl --eval-out data/eval.jsonl
```

- Training split: seed 101 (reconciliation 60% / payment_repair 25% /
  settlement 15%, ~1,500 tasks).
- Evaluation split: seed 777 (~400 tasks). Zero signature overlap is
  enforced; the script prints the contamination certificate and exits
  non-zero if overlap != 0.

## Validate the training contract

```sh
uv run python -m experiments.finetune.train_mlx --dry-run
```

Prints the frozen hyperparameters (CONFIG) and data verification without
touching a GPU.

## Canonical training run (M5 / MLX)

```sh
uv run pip install mlx-lm
uv run python -m experiments.finetune.train_mlx --data data/train.jsonl
```

Expected recipe (from the published ReconForge result): rank 16 / alpha 32 /
dropout 0.05, batch 2 with gradient checkpointing, 740 steps, loss plateau
~0.088. Stop on plateau, not on a fixed budget; a materially different run at
iteration 700 scored substantially lower, so the stopping rule matters.

## Kaggle / Unsloth track (Qwen3.8-27B)

1. Upload `data/train.jsonl` (and `data/eval.jsonl`) to a Kaggle notebook.
2. Train with Unsloth LoRA (rank 16) on Qwen3.8-27B.
3. Evaluate on seed-777 eval with the LossBench harness before publishing any
   `.eval_results/` YAML (packaging/hf/README.md).

## Rules

- Never train on the seed-777 evaluation split (contamination monitor is the
  gate).
- Publish quantized variants (MLX/GGUF) only after the canonical result is
  stable.
- Report severity-weighted recall + expected loss + calibration, never
  accuracy alone.
