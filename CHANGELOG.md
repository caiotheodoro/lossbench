# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **HF dataset card honesty (Fixes #1).** The published card is now an honest
  description of an open reference benchmark:
  - The contamination section no longer claims the eval set "cannot be
    fingerprinted from this dataset alone." That claim was false: `gold` and
    `severity` ship in plaintext in `data/eval.jsonl`. The section now states
    that the labels are public, that this is an open reference benchmark rather
    than a hidden-answer-key one, that the signature check detects training-set
    overlap and not evaluation gaming, and that third-party leaderboard
    submissions are self-reported.
  - New **Coverage** section on the card discloses that the published
    leaderboard is scored on only the first `n_tasks=60` prefix subset
    (20/domain, seed 777, trials 2, `max_steps=1`, `partial: true`) of the 1200
    shipped eval tasks (~5%), and quotes the run's `partial_note` verbatim.

### Removed

- `packaging/hf/publish.py` no longer emits `eval.yaml`. The manifest shape the
  repo produced (`id`/`dataset`/`task_types`/`metric`/`license`/`paper`) is the
  exact shape HF rejects at push-time validation (hf-publication-specs.md 4.2).
  `eval.yaml` is held until blocker B-3 (adding `lossbench` to HF's
  `evaluation_framework` enum) clears. `build_eval_yaml` / `validate_eval_yaml`
  remain for that future work.
