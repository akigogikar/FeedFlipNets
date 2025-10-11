from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

STRATS = ["bp", "dfa", "flip"]


def _fmt_mu_sigma(vals):
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    return f"{mu:.4f} ± {sd:.4f}"


def main():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from feedflipnets.core.np_mlp import train_one

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=str, default=".artifacts/bench")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for strat in STRATS:
        for s in args.seeds:
            r = train_one(strat, seed=s, steps=args.steps, lr=args.lr, batch=args.batch)
            runs.append({"strategy": strat, "seed": s, **r})
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(x) for x in runs), encoding="utf-8"
    )

    agg = {}
    for strat in STRATS:
        accs = [r["final_acc"] for r in runs if r["strategy"] == strat]
        losses = [r["final_loss"] for r in runs if r["strategy"] == strat]
        agg[strat] = {
            "n": len(accs),
            "final_acc_mu": mean(accs),
            "final_acc_sd": pstdev(accs) if len(accs) > 1 else 0.0,
            "final_loss_mu": mean(losses),
            "final_loss_sd": pstdev(losses) if len(losses) > 1 else 0.0,
        }
    bp_acc = agg["bp"]["final_acc_mu"]
    for strat in STRATS:
        agg[strat]["delta_acc_vs_bp"] = agg[strat]["final_acc_mu"] - bp_acc

    csv_path = out / "bench_micro.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "strategy",
                "seeds",
                "steps",
                "final_loss_mu",
                "final_loss_sd",
                "final_acc_mu",
                "final_acc_sd",
                "delta_acc_vs_bp",
            ]
        )
        for strat in STRATS:
            a = agg[strat]
            w.writerow(
                [
                    strat,
                    a["n"],
                    args.steps,
                    f"{a['final_loss_mu']:.4f}",
                    f"{a['final_loss_sd']:.4f}",
                    f"{a['final_acc_mu']:.4f}",
                    f"{a['final_acc_sd']:.4f}",
                    f"{a['delta_acc_vs_bp']:.4f}",
                ]
            )

    md_path = out / "bench_micro.md"
    lines = []
    lines.append("### Micro‑Benchmark: Backprop vs DFA vs Flip‑Ternary (offline)")
    lines.append("")
    seeds_line = (
        f"- Seeds: `{args.seeds}`; Steps: `{args.steps}`; "
        f"LR: `{args.lr}`; Batch: `{args.batch}`"
    )
    lines.append(seeds_line)
    lines.append("")
    lines.append(
        "| Strategy | Final Loss (μ±σ) | Final Acc (μ±σ) | ΔAcc vs BP | Seeds | Steps |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for strat in STRATS:
        fl = [r["final_loss"] for r in runs if r["strategy"] == strat]
        fa = [r["final_acc"] for r in runs if r["strategy"] == strat]
        metric_line = (
            f"| {strat.upper()} | {_fmt_mu_sigma(fl)} | {_fmt_mu_sigma(fa)} | "
            f"{agg[strat]['delta_acc_vs_bp']:+.4f} | {agg[strat]['n']} | {args.steps} |"
        )
        lines.append(metric_line)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", csv_path, md_path)


if __name__ == "__main__":
    main()
