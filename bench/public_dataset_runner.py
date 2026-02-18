#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from bench.standard_suite import anneal_solver, rc_search_solver

# Public benchmark sources (canonical references)
PUBLIC_URLS = {
    "maxcut_gset_14": "https://web.stanford.edu/~yyye/yyye/Gset/G14",
    "sat_uf20_01": "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/UF20-91/uf20-01.cnf",
    # QUBO source collections vary; this is a lightweight public reference file example format.
    "qubo_example": "https://raw.githubusercontent.com/jtiosue/qubo/master/README.md",
}


def fetch(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def parse_gset(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    n, m = map(int, txt[0].split())
    W = np.zeros((n, n), dtype=np.float64)
    for line in txt[1 : 1 + m]:
        i, j, w = line.split()
        i = int(i) - 1
        j = int(j) - 1
        w = float(w)
        W[i, j] = w
        W[j, i] = w
    return W


def cap_graph(W: np.ndarray, n_cap: int):
    n = W.shape[0]
    if n_cap <= 0 or n <= n_cap:
        return W
    idx = np.arange(n_cap)
    return W[np.ix_(idx, idx)]


def eval_maxcut(W, x):
    n = W.shape[0]
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                val += W[i, j]
    return float(val)


def main():
    ap = argparse.ArgumentParser(description="Run REZON solvers on public benchmark instances when available")
    ap.add_argument("--cache-dir", type=str, default="bench/datasets")
    ap.add_argument("--n-cap", type=int, default=120, help="cap instance size for quick benchmark")
    ap.add_argument("--out-json", type=str, default="reports/public_dataset_benchmark.json")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    rows = []

    # Max-Cut Gset
    try:
        gpath = cache / "G14.txt"
        if not gpath.exists():
            fetch(PUBLIC_URLS["maxcut_gset_14"], gpath)
        W = parse_gset(gpath)
        W = cap_graph(W, int(args.n_cap))
        n = W.shape[0]
        rng = np.random.default_rng(42)
        eval_fn = lambda x: eval_maxcut(W, x)

        no_rc = anneal_solver(n, eval_fn, rng, domain="pm1", steps=1500)
        rc = rc_search_solver(n, eval_fn, rng, domain="pm1", budget=800, use_anchor=True, use_rls=True, rc_nodes=1000, rc_warmup_steps=40)
        rows.append({"dataset": "Gset14", "problem": "maxcut", "no_rc": float(no_rc), "rc_anchor_rls": float(rc)})
    except Exception as e:
        rows.append({"dataset": "Gset14", "problem": "maxcut", "error": str(e)})

    payload = {"sources": PUBLIC_URLS, "rows": rows}
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
