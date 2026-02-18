#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def keystream_byte(key: int, idx: int) -> int:
    x = (key * 0x9E3779B185EBCA87 + (idx + 1) * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    return x & 0xFF


def encrypt(pt: bytes, key: int) -> bytes:
    return bytes([b ^ keystream_byte(key, i) for i, b in enumerate(pt)])


def key_matches(ct: bytes, known_pt: bytes, key: int) -> bool:
    for i, b in enumerate(known_pt):
        if (ct[i] ^ keystream_byte(key, i)) != b:
            return False
    return True


def brute_force(ct: bytes, known_pt: bytes, bits: int):
    attempts = 0
    for k in range(1 << bits):
        attempts += 1
        if key_matches(ct, known_pt, k):
            return k, attempts
    return None, attempts


def coarse_score(key: int, ct: bytes, known: bytes, coarse_len: int) -> float:
    n = min(coarse_len, len(known))
    err = 0.0
    for i in range(n):
        pred = ct[i] ^ keystream_byte(key, i)
        err += abs(pred - known[i]) / 255.0
    return err / max(1, n)


def full_rc_score(key: int, ct: bytes, known: bytes, precheck_len: int) -> float:
    phi = np.zeros(6, dtype=np.float64)
    phi[0] = ((key >> 0) & 0xFF) / 255.0 * 2.0 * np.pi
    phi[1] = ((key >> 8) & 0xFF) / 255.0 * 2.0 * np.pi
    phi[2] = ((key >> 16) & 0xFF) / 255.0 * 2.0 * np.pi
    K = np.array(
        [[0, 1.2, -0.6, 0.2, 0.1, -0.2],
         [1.2, 0, 0.9, -0.2, 0.0, 0.1],
         [-0.6, 0.9, 0, 0.3, -0.1, 0.2],
         [0.2, -0.2, 0.3, 0, 0.8, -0.4],
         [0.1, 0.0, -0.1, 0.8, 0, 1.0],
         [-0.2, 0.1, 0.2, -0.4, 1.0, 0]],
        dtype=np.float64,
    )
    coupling, leak, dt = 1.8, 0.02, 0.012
    anchor_amp, anchor_hz = 0.4, 200.0

    err = coarse_score(key, ct, known, precheck_len)
    t = 0.0
    for _ in range(10):
        diff = phi[None, :] - phi[:, None]
        dphi = coupling * np.sum(K * np.sin(diff), axis=1)
        dphi += anchor_amp * np.sin(2.0 * np.pi * anchor_hz * t - phi)
        dphi -= leak * phi
        dphi += (1.0 - err) * 0.25 * np.cos(phi)
        phi = (phi + dt * dphi) % (2.0 * np.pi)
        t += dt

    coherence = abs(np.mean(np.exp(1j * phi)))
    return float(err + 0.2 * coherence)


def two_stage_search(ct: bytes, known: bytes, bits: int, coarse_len: int, shortlist_ratio: float, precheck_len: int):
    keyspace = 1 << bits
    keys = np.arange(keyspace, dtype=np.int64)

    # Stage A: cheap coarse score on all keys.
    c_scores = np.empty(keyspace, dtype=np.float64)
    for i, k in enumerate(keys):
        c_scores[i] = coarse_score(int(k), ct, known, coarse_len)

    k1 = max(64, int(keyspace * shortlist_ratio))
    k1 = min(k1, keyspace)
    idx = np.argpartition(c_scores, k1 - 1)[:k1]

    # Stage B: expensive full RC score only on shortlist.
    f_scores = np.empty(k1, dtype=np.float64)
    shortlist = keys[idx]
    for i, k in enumerate(shortlist):
        f_scores[i] = full_rc_score(int(k), ct, known, precheck_len)

    order = shortlist[np.argsort(f_scores)]

    attempts = 0
    tested = set()
    for k in order:
        kk = int(k)
        tested.add(kk)
        attempts += 1
        if key_matches(ct, known, kk):
            return kk, attempts, True

    # Fallback over rest by coarse ordering
    rest = keys[np.argsort(c_scores)]
    for k in rest:
        kk = int(k)
        if kk in tested:
            continue
        attempts += 1
        if key_matches(ct, known, kk):
            return kk, attempts, False

    return None, attempts, False


def fixed_keys(keyspace: int, reps: int):
    if reps <= 1:
        qs = [0.5]
    elif reps == 2:
        qs = [0.3, 0.7]
    elif reps == 3:
        qs = [0.2, 0.5, 0.8]
    else:
        qs = [(i + 1) / (reps + 1) for i in range(reps)]
    return [min(keyspace - 1, max(0, int(q * keyspace))) for q in qs]


def run(bits_list, reps, known_len, coarse_len, shortlist_ratio, precheck_len):
    pt = (b"REZON_TWO_STAGE_BENCH_" * 6)[:96]
    known = pt[:known_len]
    rows = []
    for bits in bits_list:
        keyspace = 1 << bits
        keys = fixed_keys(keyspace, reps)
        bf_a, bf_t, ts_a, ts_t, topk_hits = [], [], [], [], 0
        for true_key in keys:
            ct = encrypt(pt, true_key)

            t0 = time.perf_counter()
            kb, ab = brute_force(ct, known, bits)
            bf_t.append(time.perf_counter() - t0)
            bf_a.append(ab)

            t1 = time.perf_counter()
            kt, at, hit = two_stage_search(ct, known, bits, coarse_len, shortlist_ratio, precheck_len)
            ts_t.append(time.perf_counter() - t1)
            ts_a.append(at)
            topk_hits += int(hit)

            if kb != true_key or kt != true_key:
                raise RuntimeError(f"mismatch bits={bits}, key={true_key}")

        bf_avg = float(np.mean(bf_a))
        ts_avg = float(np.mean(ts_a))
        rows.append({
            "bits": bits,
            "keyspace": keyspace,
            "reps": reps,
            "keys_tested": keys,
            "bruteforce_avg_attempts": bf_avg,
            "two_stage_avg_attempts": ts_avg,
            "attempt_gain_x": float(bf_avg / max(ts_avg, 1e-12)),
            "bruteforce_avg_time_s": float(np.mean(bf_t)),
            "two_stage_avg_time_s": float(np.mean(ts_t)),
            "time_gain_x": float(np.mean(bf_t) / max(np.mean(ts_t), 1e-12)),
            "shortlist_hit_rate": float(topk_hits / max(1, reps)),
            "grover_ideal_queries": float(math.sqrt(keyspace)),
        })
    return rows


def to_md(rows):
    lines = [
        "# Two-Stage Phase-Guided Benchmark",
        "",
        "| bits | bf attempts | two-stage attempts | attempt gain | bf time (s) | two-stage time (s) | time gain | shortlist hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bits']} | {r['bruteforce_avg_attempts']:.2f} | {r['two_stage_avg_attempts']:.2f} | {r['attempt_gain_x']:.3f} | "
            f"{r['bruteforce_avg_time_s']:.6f} | {r['two_stage_avg_time_s']:.6f} | {r['time_gain_x']:.3f} | {r['shortlist_hit_rate']:.3f} |"
        )
    lines.append("")
    lines.append("Two-stage = cheap coarse score on all keys, expensive RC score only on shortlist.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Two-stage phase-guided key ranking benchmark")
    ap.add_argument("--bits", type=str, default="8,10,12,14,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--known-len", type=int, default=10)
    ap.add_argument("--coarse-len", type=int, default=2)
    ap.add_argument("--precheck-len", type=int, default=4)
    ap.add_argument("--shortlist-ratio", type=float, default=0.02)
    ap.add_argument("--out-json", type=str, default="reports/two_stage_phase_guided.json")
    ap.add_argument("--out-md", type=str, default="reports/two_stage_phase_guided.md")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    rows = run(bits_list, args.reps, args.known_len, args.coarse_len, args.shortlist_ratio, args.precheck_len)
    payload = {
        "config": {
            "bits": bits_list,
            "reps": int(args.reps),
            "known_len": int(args.known_len),
            "coarse_len": int(args.coarse_len),
            "precheck_len": int(args.precheck_len),
            "shortlist_ratio": float(args.shortlist_ratio),
        },
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_md(rows), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
