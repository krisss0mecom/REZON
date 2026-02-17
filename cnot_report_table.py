#!/usr/bin/env python3
import argparse
import json


def main():
    p = argparse.ArgumentParser(description="Render markdown table from cnot_rls JSON report.")
    p.add_argument("--in-json", type=str, default="results/cnot_rls_report.json")
    p.add_argument("--out-md", type=str, default="results/cnot_rls_table.md")
    args = p.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("details", [])
    lines = [
        "| Control | Target | Pred score | Pred bit | Expected | OK |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['control']} | {r['target']} | {r['pred_score']:+.4f} | {r['pred_bit']} | {r['expected_bit']} | {r['ok']} |"
        )
    lines.append("")
    lines.append(f"**Score:** {data.get('score_4', 0)}/4  ")
    lines.append(f"**Accuracy:** {data.get('accuracy', 0.0):.4f}")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
