## ⚠️ STUB PIPELINE SMOKE OUTPUT

This card is **not** a LossBench result. The runner is gold-keyed by task id, so every model scores perfectly by construction. Set LOSSBENCH_API_KEY for real inference.

# deepseek-v4-flash — LossBench finance-v1

| Metric | Value |
|---|---|
| pass@1 | 1.000 |
| pass^k | 1.000 |
| false-success rate | n/a — self-verifying harness (see issue #10) |
| severity-weighted loss | 0.0000 |
| total cost | 0.0000 |

Evaluation: LossBench finance-v1 (reconciliation / payment_repair /
settlement), seed 777, verifier-as-oracle. Severity costs are
pluggable inputs; see the severity-cost registry.

Run: `uv run python -m scripts.full_run`
