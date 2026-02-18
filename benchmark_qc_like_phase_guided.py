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


def phase_guided_score(key: int, ct: bytes, known_pt: bytes, precheck_len: int) -> float:
    # RC-inspired analog score:
    # 1) inject residual error signal from short known prefix
    # 2) evolve simple coupled phase state
    # 3) use final coherence error as rank score (lower is better)
    phi_a = (key & 0xFF) / 255.0 * 2.0 * np.pi
    phi_b = ((key >> 8) & 0xFF) / 255.0 * 2.0 * np.pi
    coupling = 1.9
    leak = 0.02
    dt = 0.015
    anchor_hz = 200.0
    anchor_amp = 0.4

    err_acc = 0.0
    n = min(precheck_len, len(known_pt))
    for i in range(n):
        pred = ct[i] ^ keystream_byte(key, i)
        err_acc += abs(pred - known_pt[i]) / 255.0
    err = err_acc / max(1, n)

    # 8 micro-steps of phase evolution from error drive
    t = 0.0
    for _ in range(8):
        d = np.sin(phi_b - phi_a)
        drive = (1.0 - err) * 1.5
        dphi_a = coupling * d + anchor_amp * np.sin(2.0 * np.pi * anchor_hz * t - phi_a) - leak * phi_a
        dphi_b = -coupling * d + drive * np.cos(phi_b) + anchor_amp * np.sin(2.0 * np.pi * anchor_hz * t - phi_b) - leak * phi_b
        phi_a = (phi_a + dt * dphi_a) % (2.0 * np.pi)
        phi_b = (phi_b + dt * dphi_b) % (2.0 * np.pi)
        t += dt

    # Favor low error + coherent anti-phase structure.
    coherence = abs(np.cos(phi_a - phi_b))
    return float(err + 0.15 * coherence)


def phase_guided_search(ct: bytes, known_pt: bytes, bits: int, precheck_len: int):
    keyspace = 1 << bits
    keys = np.arange(keyspace, dtype=np.int64)
    scores = np.empty(keyspace, dtype=np.float64)
    for i, k in enumerate(keys):
        scores[i] = phase_guided_score(int(k), ct, known_pt, precheck_len)
    order = keys[np.argsort(scores)]

    attempts = 0
    for k in order:
        attempts += 1
        if key_matches(ct, known_pt, int(k)):
            return int(k), attempts
    return None, attempts


def make_fixed_keys(keyspace: int, reps: int):
    if reps <= 1:
        qs = [0.5]
    elif reps == 2:
        qs = [0.30, 0.70]
    elif reps == 3:
        qs = [0.20, 0.50, 0.80]
    else:
        qs = [(i + 1) / (reps + 1) for i in range(reps)]
    return [min(keyspace - 1, max(0, int(q * keyspace))) for q in qs]


def reps_for_bits(bits: int) -> int:
    if bits <= 10:
        return 6
    if bits <= 12:
        return 4
    if bits <= 14:
        return 3
    return 2


def run(bits_list, known_len: int, precheck_len: int):
    pt = (b"REZON_PHASE_GUIDED_DECRYPT_BENCH_" * 3)[:96]
    known = pt[:known_len]
    rows = []
    for bits in bits_list:
        keyspace = 1 << bits
        reps = reps_for_bits(bits)
        keys = make_fixed_keys(keyspace, reps)

        bf_attempts, bf_time = [], []
        pg_attempts, pg_time = [], []
        for true_key in keys:
            ct = encrypt(pt, true_key)

            t0 = time.perf_counter()
            found_bf, a_bf = brute_force(ct, known, bits)
            bf_time.append(time.perf_counter() - t0)
            bf_attempts.append(a_bf)

            t1 = time.perf_counter()
            found_pg, a_pg = phase_guided_search(ct, known, bits, precheck_len=precheck_len)
            pg_time.append(time.perf_counter() - t1)
            pg_attempts.append(a_pg)

            if found_bf != true_key or found_pg != true_key:
                raise RuntimeError(f"key recovery mismatch for bits={bits}, key={true_key}")

        bf_avg = float(np.mean(bf_attempts))
        pg_avg = float(np.mean(pg_attempts))
        rows.append(
            {
                "bits": bits,
                "keyspace": keyspace,
                "reps": reps,
                "keys_tested": keys,
                "bruteforce_avg_attempts": bf_avg,
                "phase_guided_avg_attempts": pg_avg,
                "attempt_gain_x": float(bf_avg / max(pg_avg, 1e-9)),
                "bruteforce_avg_time_s": float(np.mean(bf_time)),
                "phase_guided_avg_time_s": float(np.mean(pg_time)),
                "time_gain_x": float(np.mean(bf_time) / max(np.mean(pg_time), 1e-12)),
                "grover_ideal_queries": float(math.sqrt(keyspace)),
            }
        )
    return rows


def to_markdown(rows):
    out = []
    out.append("# Phase-Guided Decryption Benchmark")
    out.append("")
    out.append("| bits | keyspace | bf avg attempts | phase-guided avg attempts | attempts gain (bf/pg) | bf avg time (s) | pg avg time (s) | time gain (bf/pg) | Grover ref queries |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['bits']} | {r['keyspace']} | {r['bruteforce_avg_attempts']:.2f} | {r['phase_guided_avg_attempts']:.2f} | "
            f"{r['attempt_gain_x']:.3f} | {r['bruteforce_avg_time_s']:.6f} | {r['phase_guided_avg_time_s']:.6f} | "
            f"{r['time_gain_x']:.3f} | {r['grover_ideal_queries']:.2f} |"
        )
    out.append("")
    out.append("Notes:")
    out.append("- Phase-guided search reorders candidates using RC-inspired analog scoring.")
    out.append("- Gains in attempts do not guarantee runtime gains due to ranking overhead.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Benchmark phase-guided candidate ordering vs brute-force.")
    ap.add_argument("--bits", type=str, default="8,10,12,14,16")
    ap.add_argument("--known-len", type=int, default=10)
    ap.add_argument("--precheck-len", type=int, default=4)
    ap.add_argument("--out-json", type=str, default="reports/qc_like_phase_guided_benchmark.json")
    ap.add_argument("--out-md", type=str, default="reports/qc_like_phase_guided_benchmark.md")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    rows = run(bits_list, known_len=args.known_len, precheck_len=args.precheck_len)
    payload = {
        "config": {
            "bits": bits_list,
            "known_len": int(args.known_len),
            "precheck_len": int(args.precheck_len),
        },
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md = Path(args.out_md)
    out_md.write_text(to_markdown(rows), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
