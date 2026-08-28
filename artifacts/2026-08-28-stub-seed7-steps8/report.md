# LossBench Report

## Metadata

- cost_model: reconciliation

- generated_at: 2026-08-28T00:41:36.880817+00:00

- suite: finance-v1

## Losses

| Model | Loss |
| --- | --- |
| deepseek-v4-flash | 0.0000 |
| nemotron-3.5-lightning | 0.0000 |
| qwen3.8-27b | 0.0000 |
| reconforge-1.7b | 0.0000 |

## Cost Sensitivity: deepseek-v4-flash

| ratio | loss |
| --- | --- |
| 1.0000 | 218.4000 |
| 2.0000 | 428.4000 |
| 5.0000 | 1058.4000 |
| 10.0000 | 2108.4000 |
| 100.0000 | 21008.4000 |

## Cost Sensitivity: nemotron-3.5-lightning

| ratio | loss |
| --- | --- |
| 1.0000 | 218.4000 |
| 2.0000 | 428.4000 |
| 5.0000 | 1058.4000 |
| 10.0000 | 2108.4000 |
| 100.0000 | 21008.4000 |

## Cost Sensitivity: qwen3.8-27b

| ratio | loss |
| --- | --- |
| 1.0000 | 218.4000 |
| 2.0000 | 428.4000 |
| 5.0000 | 1058.4000 |
| 10.0000 | 2108.4000 |
| 100.0000 | 21008.4000 |

## Cost Sensitivity: reconforge-1.7b

| ratio | loss |
| --- | --- |
| 1.0000 | 218.4000 |
| 2.0000 | 428.4000 |
| 5.0000 | 1058.4000 |
| 10.0000 | 2108.4000 |
| 100.0000 | 21008.4000 |

## Calibration

| Model | ECE | n |
| --- | --- | --- |
| deepseek-v4-flash | 0.1000 | 900 |
| nemotron-3.5-lightning | 0.1000 | 900 |
| qwen3.8-27b | 0.1000 | 900 |
| reconforge-1.7b | 0.1000 | 900 |

## Honest Limits

1. synthetic data only; severity costs are contested, sourced inputs

2. STUB RUN: the runner is gold-keyed by task id, so every model scores perfectly by construction. These are not model results. Set LOSSBENCH_API_KEY for real inference.

3. sensitivity curves use synthetic per-model error patterns as a metric demonstration, not model measurements
