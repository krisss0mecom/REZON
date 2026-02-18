#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_points(report_dir: Path, seeds: int):
    points = []
    for n in (80, 160, 320, 640, 1000):
        p = report_dir / f"phase_entanglement_nodes{n}_s{seeds}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        stats = data.get("stats", {})
        points.append(
            {
                "nodes": n,
                "quality": float(stats.get("quality", {}).get("mean", float("nan"))),
                "quality_vs_ls": float(stats.get("quality_vs_ls_baseline", {}).get("mean", float("nan"))),
            }
        )
    return points


def write_md(points, out_md: Path):
    lines = [
        "# Quality vs Nodes",
        "",
        "| nodes | quality_vs_simple_baseline | quality_vs_ls_baseline |",
        "|---:|---:|---:|",
    ]
    for r in points:
        lines.append(f"| {r['nodes']} | {r['quality']:.6f} | {r['quality_vs_ls']:.6f} |")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_png(points, out_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    x = [r["nodes"] for r in points]
    y1 = [r["quality"] for r in points]
    y2 = [r["quality_vs_ls"] for r in points]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y1, marker="o", label="quality vs simple baseline")
    plt.plot(x, y2, marker="s", label="quality vs LS baseline")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])
    plt.xlabel("nodes")
    plt.ylabel("ratio")
    plt.title("Phase-Guided Max-Cut Quality by Node Count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    return True


def main():
    ap = argparse.ArgumentParser(description="Plot quality ratios vs node count from correlation pack reports")
    ap.add_argument("--report-dir", type=str, default="reports/correlation_pack")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out-md", type=str, default="reports/quality_vs_nodes_s8.md")
    ap.add_argument("--out-png", type=str, default="reports/quality_vs_nodes_s8.png")
    args = ap.parse_args()

    points = load_points(Path(args.report_dir), args.seeds)
    if not points:
        raise SystemExit("No matching correlation-pack reports found")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_md(points, out_md)
    png_ok = write_png(points, Path(args.out_png))

    print(f"saved: {out_md}")
    if png_ok:
        print(f"saved: {args.out_png}")
    else:
        print("plot skipped: matplotlib not available")


if __name__ == "__main__":
    main()

