"""Shim package that exposes the upstream ``importlinter`` distribution."""
from __future__ import annotations

import sys
from importlib import metadata, util
from pathlib import Path
from types import ModuleType
from typing import Iterable

_BASE_PATH = Path(__file__).parent.resolve()
_VENDOR_NAME = "_importlinter_vendor"
_VENDOR_MODULE: ModuleType | None = None
_VENDOR_PATH: Path | None = None


def _load_vendor() -> ModuleType:
    global _VENDOR_MODULE, _VENDOR_PATH
    if _VENDOR_MODULE is not None:
        return _VENDOR_MODULE

    dist = metadata.distribution("import-linter")
    package_path = Path(dist.locate_file("importlinter"))
    globals()["__path__"] = [str(_BASE_PATH), str(package_path)]

    spec = util.spec_from_file_location(
        _VENDOR_NAME,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate importlinter vendor package")

    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    _VENDOR_MODULE = module
    _VENDOR_PATH = package_path
    return module


def _export_vendor_attributes(module: ModuleType) -> None:
    public_names: Iterable[str]
    public_names = getattr(module, "__all__", None) or (
        name for name in dir(module) if not name.startswith("_")
    )

    globals().update({name: getattr(module, name) for name in public_names})
    globals()["__all__"] = list(public_names)


def _vendor_path() -> Path:
    if _VENDOR_PATH is None:
        _load_vendor()
        assert _VENDOR_PATH is not None
    return _VENDOR_PATH


_vendor = _load_vendor()
_export_vendor_attributes(_vendor)
