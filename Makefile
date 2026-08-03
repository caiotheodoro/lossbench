.PHONY: validate lint test install

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

validate: lint test
