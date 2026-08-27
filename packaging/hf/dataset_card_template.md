---
license: $license_name
pretty_name: $benchmark_id
language:
  - en
task_categories:
  - text-generation
size_categories:
  - 1K<n<10K
tags:
  - finance
  - back-office
  - agents
  - expected-loss
  - severity-weighted
  - verifier-as-oracle
  - structured-outputs
configs:
  - config_name: default
    default: true
    data_files:
      - split: eval
        path: data/eval.jsonl
      - split: train
        path: data/train.jsonl
---

# $benchmark_id

$description

## Overview

| | |
|---|---|
| Task count | $task_count |
| Domains | $domains |
| License | $license_name |

## Tasks

Each task is an agentic back-office scenario with a deterministic seed, an
executable verifier, and a gold outcome. Tasks are grouped by domain; every
task carries a business severity band, a difficulty score, and a reference
cost model. Models are ranked by expected loss under a cost model, not by raw
accuracy alone.

## License

This dataset is released under the $license_name license.

## Severity taxonomy

Every task is labeled with exactly one severity band:

$severity_bullets

Severity bands describe the business impact of failure and are assigned at
generation time, never inferred from model outputs.

## Cost models

Severity costs are **pluggable inputs**, not constants of the dataset. Each
cost model maps severity bands to business error costs K(σ); the dataset ships
reference cost models ($cost_model_ids) and adopters may register their own.
Loss numbers therefore depend on the chosen cost model, and conclusions must
be shown across a K range — sweeping the severity-cost scale as well as the
pass@k window — so that rankings do not silently depend on one arbitrary
operating point.

## Results

$results_table

## Coverage

$coverage_note

## Honest limits

$honest_limits

## Contamination policy

$contamination_policy

## Reproducibility

$reproducibility_notes

## Contact

$contact
