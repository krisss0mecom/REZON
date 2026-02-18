#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Use C++ nth_element top-K ranker for RC candidate scores")
    ap.add_argument("--keys", type=int, default=65536)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--out", type=str, default="reports/topk_ranker_smoke.txt")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cpp = root / "topk_ranker.cpp"
    binp = root / "topk_ranker"

    subprocess.check_call(["g++", "-O3", "-std=c++17", str(cpp), "-o", str(binp)])

    rng = np.random.default_rng(123)
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "scores.txt"
        with inp.open("w", encoding="utf-8") as f:
            for i in range(args.keys):
                score = float(rng.random())
                f.write(f"{i} {score}\n")
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([str(binp), str(inp), str(args.k), str(outp)])

    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
