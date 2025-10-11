.PHONY: setup test smoke format lint run precommit clean

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
RUFF := $(VENV)/bin/ruff
FLAKE8 := $(VENV)/bin/flake8
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

SETUP_PATHS := cli feedflipnets scripts tests
SMOKE_PRESETS := mnist_mlp_dfa ucr_gunpoint_mlp_dfa california_housing_mlp_dfa 20newsgroups_bow_mlp_dfa

setup:
	@[ -d $(VENV) ] || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PRECOMMIT) install

format:
	$(ISORT) $(SETUP_PATHS)
	$(BLACK) $(SETUP_PATHS)

lint:
	$(RUFF) check $(SETUP_PATHS)
	$(FLAKE8) $(SETUP_PATHS)
	$(BLACK) --check $(SETUP_PATHS)

precommit:
	$(PRECOMMIT) run --all-files

test:
	FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTEST) -q

smoke:
	@for preset in $(SMOKE_PRESETS); do \
		echo "==> $$preset"; \
		FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTHON_BIN) -m cli.main --preset $$preset --offline; \
	done

run:
	@if [ -z "$(PRESET)" ]; then \
		echo "Usage: make run PRESET=<preset> [EXTRA_ARGS='--feedback dfa']"; \
		exit 1; \
	fi
	FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTHON_BIN) -m cli.main --preset $(PRESET) $(EXTRA_ARGS)

clean:
	rm -rf $(VENV)
