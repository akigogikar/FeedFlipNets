"""Command line entry point for FeedFlipNets experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

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
    parser.add_argument("--preset", choices=preset_names, help="Which preset configuration to run")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--steps", type=int, help="Override the number of training steps")
    parser.add_argument("--run-dir", type=str, help="Override the run directory")
    parser.add_argument("--dump-config", type=Path, help="Dump the resolved config to a JSON file")
    args = parser.parse_args(argv)
    if args.list_presets:
        for name in preset_names:
            print(name)
        raise SystemExit(0)
    if not args.preset:
        parser.error("--preset is required unless --list-presets is used")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = pipelines.load_preset(args.preset)
    config = json.loads(json.dumps(config))  # deep copy via JSON for safety

    if args.steps is not None:
        config.setdefault("train", {})["steps"] = args.steps
    if args.run_dir is not None:
        config.setdefault("train", {})["run_dir"] = args.run_dir

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

