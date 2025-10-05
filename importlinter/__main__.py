"""Shim entry point so ``python -m importlinter`` works."""
from __future__ import annotations

import sys
from importlib import import_module

from . import _VENDOR_NAME, _load_vendor


def main() -> None:
    _load_vendor()
    if len(sys.argv) > 1 and sys.argv[1] == "lint":
        sys.argv.pop(1)
    cli_module = import_module(f"{_VENDOR_NAME}.cli")
    cli_module.lint_imports_command.main(prog_name="importlinter")


if __name__ == "__main__":
    main()
