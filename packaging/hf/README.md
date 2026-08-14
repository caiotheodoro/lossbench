# LossBench — HF Community Evals packaging (P1.18)

This directory holds the release artifacts that register LossBench as a
Hugging Face dataset with **Community Evals**: the `eval.yaml` registration
manifest, the dataset card template, and the programmatic builders in
`exporter.py`.

## What each file is for

| File | Purpose |
|---|---|
| `eval.yaml` | Registration manifest consumed by the Hugging Face Hub |
| `dataset_card_template.md` | Card template; render it with `build_dataset_card` and commit as the dataset repo's `README.md` |
| `exporter.py` | `build_eval_yaml`, `build_dataset_card`, `validate_eval_yaml`, `tasks_to_jsonl` |
| `README.md` | This file: publish procedure |

## Dataset repo layout

Create a dataset repo named `LossBench/lossbench` (own org: `OWNER/lossbench`)
with this layout:

```
lossbench/
├── eval.yaml              # registration manifest (this file, root)
├── README.md              # rendered dataset card
├── data/
│   ├── eval.jsonl         # canonical eval-set export
│   └── train.jsonl        # fine-tune split
└── results/
    └── eval-results.json  # scores written by community eval runs
```

Publish procedure:

1. `huggingface-cli login`
2. `git clone https://huggingface.co/datasets/LossBench/lossbench`
3. Copy `eval.yaml` to the repo root and the rendered card to `README.md`.
4. Export tasks with `tasks_to_jsonl(tasks, "data/eval.jsonl")` and commit.
5. `huggingface-cli upload LossBench/lossbench . .`

## How eval.yaml registers the benchmark

The Hub treats a root-level `eval.yaml` in a dataset repo as a Community Evals
registration. `build_eval_yaml` emits the documented shape:

```python
{
    "id", "description", "version",
    "dataset": {"path", "revision"},
    "task_types",
    "metric": {"name", "higher_is_better"},
    "license",
    "paper": {"title", "url"} | None,
}
```

Contract notes:

- `metric.name` must be `severity_weighted_loss` and
  `metric.higher_is_better` must be `false` (loss is minimized). The eval
  harness must compute this metric on the same severity/cost model used for
  scoring.
- `dataset.revision` is `main` here; pin a commit SHA in a release to make
  runs reproducible.
- Run `validate_eval_yaml` in CI before pushing the manifest.

## How scores reach model cards (.eval_results)

A model card that includes LossBench among its Community Evals gets scored by
the community eval runner. The runner:

1. Loads `data/eval.jsonl` from the dataset repo at the pinned revision.
2. Executes each task under the harness (verifier-as-oracle, pass^k) and
   computes `severity_weighted_loss` under the reference cost models.
3. Writes `results/eval-results.json` into the run output.
4. Attaches the results to the model card via the `.eval_results` entry, so
   `severity_weighted_loss` appears on the model card next to standard
   accuracy metrics.

Because severity costs are **pluggable inputs**, every result set must report
loss across a K range (severity-cost scale and pass@k window), not at a single
arbitrary operating point.

## Contamination and training exports

`tasks_to_jsonl` serializes via `model_dump_json` but **never writes the
`signature` field** (the eval-set identity hash). Use it for training splits
only; eval-set signatures must not appear in any artifact an adopter could
fine-tune on. Eval sets are screened by the contamination monitor before each
release.
