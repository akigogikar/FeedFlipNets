"""Generate a paper-ready bundle from a FeedFlipNets run directory."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

from feedflipnets.reporting.summary import write_summary


def _load_metrics(path: Path) -> Iterable[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_tables(summary: dict[str, object], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return
    headers = ["metric", "min", "max", "mean", "last", "tail_auc"]
    rows = [headers]
    for name in sorted(metrics):
        stats = metrics[name]
        if not isinstance(stats, dict):
            continue
        rows.append(
            [
                name,
                f"{stats.get('min', 0.0):.6f}",
                f"{stats.get('max', 0.0):.6f}",
                f"{stats.get('mean', 0.0):.6f}",
                f"{stats.get('last', 0.0):.6f}",
                f"{stats.get('tail_auc', 0.0):.6f}",
            ]
        )
    table_path = tables_dir / "metrics_summary.csv"
    table_path.write_text("\n".join(",".join(row) for row in rows))


def _write_methods_stub(
    path: Path, manifest: dict[str, object], summary: dict[str, object]
) -> None:
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    dataset = manifest.get("dataset", {}) if isinstance(manifest, dict) else {}
    model = config.get("model", {}) if isinstance(config, dict) else {}
    train = config.get("train", {}) if isinstance(config, dict) else {}
    run_dir = train.get("run_dir", "") if isinstance(train, dict) else ""
    run_id = Path(run_dir).name if run_dir else ""
    tail_window = summary.get("tail_window", 0)
    dataset_name = (
        dataset.get("type", dataset.get("name", "unknown"))
        if isinstance(dataset, dict)
        else "unknown"
    )
    strategy = model.get("strategy", "unknown") if isinstance(model, dict) else "unknown"
    seed = train.get("seed", "unknown") if isinstance(train, dict) else "unknown"

    lines = [
        "# Methods",
        "",
        f"- **Run directory**: {run_dir or 'unknown'}",
        f"- **Run ID**: {run_id or 'unknown'}",
        f"- **Dataset**: {dataset_name}",
        f"- **Strategy**: {strategy}",
        f"- **Steps logged**: {summary.get('records', 0)}",
        f"- **Tail window**: {tail_window}",
        f"- **Seed**: {seed}",
        "",
        "This bundle was generated with deterministic settings for reproducibility.",
    ]
    path.write_text("\n".join(lines))


def _generate_plots(metrics: Iterable[dict[str, object]], figures_dir: Path) -> None:
    from feedflipnets.reporting.plots import PlotAdapter

    adapter = PlotAdapter(figures_dir, enable_plots=True)
    for record in metrics:
        step = int(record.get("step", 0))
        adapter.on_step(step, record)
    adapter.close()


def _create_zip(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w") as bundle:
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = str(path.relative_to(out_dir.parent))
            info = zipfile.ZipInfo(arcname)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    return zip_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory to bundle")
    parser.add_argument(
        "--out",
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("paper_bundle"),
        help="Output directory for the bundle",
    )
    parser.add_argument("--include-plots", action="store_true", help="Generate headless plots")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_src = run_dir / "metrics.jsonl"
    if not metrics_src.exists():
        raise FileNotFoundError(f"metrics.jsonl not found in {run_dir}")
    metrics_dest = out_dir / "metrics.jsonl"
    metrics_dest.write_bytes(metrics_src.read_bytes())

    summary_dest = out_dir / "summary.json"
    summary_path = write_summary(metrics_dest, summary_dest)
    summary = json.loads(Path(summary_path).read_text())

    manifest_src = run_dir / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text())
        manifest_out = out_dir / "manifest.json"
        manifest_out.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    else:
        (out_dir / "manifest.json").write_text(json.dumps({}, indent=2))

    _write_tables(summary, tables_dir)

    methods_path = out_dir / "methods.md"
    _write_methods_stub(methods_path, manifest, summary)

    metrics_records = _load_metrics(metrics_dest)
    if args.include_plots:
        _generate_plots(metrics_records, figures_dir)

    zip_path = _create_zip(out_dir)
    print(json.dumps({"bundle_dir": str(out_dir), "bundle_zip": str(zip_path)}))


if __name__ == "__main__":
    main()
