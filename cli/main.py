"""Command line entry point for FeedFlipNets experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from feedflipnets.training import pipelines


def _format_result(result) -> str:
    payload = {
        "steps": result.steps,
        "metrics": result.metrics_path,
        "manifest": result.manifest_path,
    }
    return json.dumps(payload)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    preset_names = sorted(pipelines.presets().keys())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=preset_names,
        default="basic_dfa_cpu",
        help="Preset configuration to execute",
    )
    parser.add_argument(
        "--config", type=Path, help="Optional JSON/YAML config override"
    )
    parser.add_argument(
        "--enable-plots", action="store_true", help="Enable plotting adapters"
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force offline dataset usage",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets and exit"
    )
    parser.add_argument(
        "--dump-config", type=Path, help="Dump the resolved config to a JSON file"
    )
    return parser.parse_args(argv)


def _load_override(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load YAML configs") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_presets:
        for name in sorted(pipelines.presets().keys()):
            print(name)
        raise SystemExit(0)

    config = pipelines.load_preset(args.preset)
    config = json.loads(json.dumps(config))

    if args.config:
        override = _load_override(args.config)
        config = _merge(config, override)

    if args.enable_plots:
        config.setdefault("train", {})["enable_plots"] = True

    config["offline"] = bool(args.offline)

    os.environ["FEEDFLIP_DATA_OFFLINE"] = "1" if args.offline else "0"

    if args.dump_config:
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(json.dumps(config, indent=2))

    result = pipelines.run_pipeline(config)

    if isinstance(result, list):
        for item in result:
            print(_format_result(item))
    else:
        print(_format_result(result))


if __name__ == "__main__":
    main()
