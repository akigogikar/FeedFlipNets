.PHONY: setup smoke test lint

setup:
	pip install -r requirements-lock.txt

smoke:
	FEEDFLIP_DATA_OFFLINE=1 python -m cli.main --preset synthetic-min

test:
	FEEDFLIP_DATA_OFFLINE=1 pytest

lint:
	lint-imports
