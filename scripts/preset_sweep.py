"""Utility for running small hyper-parameter sweeps over presets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

from feedflipnets.training import pipelines


def _parse_int_list(values: Sequence[str]) -> list[int]:
    return [int(v) for v in values]


def _parse_float_list(values: Sequence[str]) -> list[float]:
    return [float(v) for v in values]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, help="Preset name to sweep over")
    parser.add_argument(
        "--feedback",
        nargs="+",
        default=["backprop", "dfa", "ternary_dfa"],
        help="Feedback strategies to evaluate",
    )
    parser.add_argument(
        "--ternary",
        nargs="+",
        default=["off", "per_step"],
        help="Ternary schedules to evaluate",
    )
    parser.add_argument(
        "--lr",
        nargs="+",
        default=["0.1", "0.01"],
        help="Learning rates to evaluate",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        default=["128", "256"],
        help="Hidden layer sizes to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/sweeps"),
        help="Directory to store sweep run outputs",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow datasets to download instead of using offline fixtures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configurations without executing them",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    base = pipelines.load_preset(args.preset)
    base_model = dict(base.get("model", {}))
    base_hidden = list(base_model.get("hidden", []))
    depth = len(base_hidden) if base_hidden else 1

    feedback_values = [str(v) for v in args.feedback]
    ternary_values = [str(v) for v in args.ternary]
    lr_values = _parse_float_list(args.lr)
    hidden_values = _parse_int_list(args.hidden)

    output_root = Path(args.output_dir) / args.preset
    output_root.mkdir(parents=True, exist_ok=True)

    combinations = itertools.product(feedback_values, ternary_values, lr_values, hidden_values)

    for feedback, ternary, lr, hidden in combinations:
        config = json.loads(json.dumps(base))
        config["offline"] = not args.online

        model_cfg = config.setdefault("model", {})
        train_cfg = config.setdefault("train", {})

        model_cfg["strategy"] = feedback
        layer_widths = [hidden for _ in range(max(1, depth))]
        model_cfg["hidden"] = layer_widths

        resolved_ternary = "off" if feedback == "backprop" else ternary
        train_cfg["ternary"] = resolved_ternary
        train_cfg["lr"] = float(lr)

        run_name = f"feedback-{feedback}_ternary-{resolved_ternary}_lr-{lr:g}_hidden-{hidden}"
        run_dir = output_root / run_name.replace(".", "p")
        train_cfg["run_dir"] = str(run_dir)

        payload = {
            "preset": args.preset,
            "feedback": feedback,
            "ternary": resolved_ternary,
            "lr": lr,
            "hidden": layer_widths,
            "run_dir": str(run_dir),
        }

        if args.dry_run:
            print(json.dumps({"run": run_name, "config": payload}, sort_keys=True))
            continue

        result = pipelines.run_pipeline(config)
        result_payload = {
            "run": run_name,
            "metrics": result.metrics_path,
            "manifest": result.manifest_path,
        }
        if getattr(result, "summary_path", None):
            result_payload["summary"] = result.summary_path
        print(json.dumps(result_payload, sort_keys=True))


if __name__ == "__main__":
    main()
