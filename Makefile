.PHONY: validate lint test install determinism

install:
	uv sync --dev
	@# Hardened python>=3.11.14 (uv-managed builds) skips .pth files carrying
	@# the macOS hidden flag that uv sets, silently breaking editable installs.
	@# Inject the src dir via sitecustomize.py instead: processed at startup,
	@# immune to uv re-syncs rewriting .pth files.
	@printf '%s\n' 'import os, sys' '' '_src = "$(CURDIR)/src"' \
	  'if os.path.isdir(_src) and _src not in sys.path:' \
	  '    sys.path.insert(0, _src)' \
	  > .venv/lib/python3.11/site-packages/sitecustomize.py

lint:
	uv run ruff check src tests

test:
	uv run pytest

determinism:
	@# Two full runs from the same seed must produce byte-identical report.md
	@# and contamination certificates (the determinism gate the design
	@# requires). Runtime metadata (generated_at, durations) is excluded.
	rm -rf /tmp/lb-golden-a /tmp/lb-golden-b
	uv run python -m scripts.full_run --out /tmp/lb-golden-a --seed 7 --n-tasks 100
	uv run python -m scripts.full_run --out /tmp/lb-golden-b --seed 7 --n-tasks 100
	uv run python -m scripts.check_determinism

validate: lint test
