#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path


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


def brute_force_key(ct: bytes, known_pt: bytes, bits: int):
    max_k = 1 << bits
    attempts = 0
    for k in range(max_k):
        attempts += 1
        if key_matches(ct, known_pt, k):
            return k, attempts
    return None, attempts


def reps_for_bits(bits: int) -> int:
    if bits <= 12:
        return 20
    if bits <= 16:
        return 10
    if bits <= 20:
        return 4
    return 2


def make_fixed_keys(keyspace: int, reps: int):
    # Deterministic key positions for reproducible scaling curves.
    if reps <= 1:
        qs = [0.5]
    elif reps == 2:
        qs = [0.30, 0.70]
    elif reps == 3:
        qs = [0.20, 0.50, 0.80]
    elif reps == 4:
        qs = [0.15, 0.35, 0.65, 0.85]
    else:
        qs = [(i + 1) / (reps + 1) for i in range(reps)]
    keys = [min(keyspace - 1, max(0, int(q * keyspace))) for q in qs]
    return keys


def run_sweep(bits_list, known_len: int):
    pt = (b"REZON_QC_LIKE_DECRYPT_BENCHMARK_" * 3)[:96]
    known = pt[:known_len]
    rows = []
    for bits in bits_list:
        keyspace = 1 << bits
        reps = reps_for_bits(bits)
        keys = make_fixed_keys(keyspace, reps)
        attempts = []
        times = []
        for true_key in keys:
            ct = encrypt(pt, true_key)
            t0 = time.perf_counter()
            found, a = brute_force_key(ct, known, bits)
            dt = time.perf_counter() - t0
            attempts.append(a)
            times.append(dt)
            if found != true_key:
                raise RuntimeError(f"Incorrect key recovery for bits={bits}")
        avg_attempts = sum(attempts) / len(attempts)
        avg_time = sum(times) / len(times)
        grover = math.sqrt(keyspace)
        rows.append(
            {
                "bits": bits,
                "keyspace": keyspace,
                "reps": reps,
                "keys_tested": keys,
                "classical_avg_attempts": avg_attempts,
                "classical_avg_time_s": avg_time,
                "grover_ideal_queries": grover,
                "query_ratio_classical_to_grover": avg_attempts / grover,
            }
        )
    return rows


def to_markdown(rows):
    out = []
    out.append("# QC-like Decryption Benchmark")
    out.append("")
    out.append("| key bits | keyspace | classical avg attempts | classical avg time (s) | Grover ideal queries | classical/Grover |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['bits']} | {r['keyspace']} | {r['classical_avg_attempts']:.2f} | "
            f"{r['classical_avg_time_s']:.6f} | {r['grover_ideal_queries']:.2f} | "
            f"{r['query_ratio_classical_to_grover']:.2f} |"
        )
    out.append("")
    out.append("Notes:")
    out.append("- Classical column is real brute-force runtime/query count on this machine.")
    out.append("- Grover column is a theoretical query lower bound reference (not measured hardware QC runtime).")
    out.append("- Keys are chosen at fixed quantiles of keyspace for reproducible scaling.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="QC-like decryption benchmark: brute-force key search vs Grover reference.")
    ap.add_argument("--bits", type=str, default="8,10,12,14,16,18,20,22,24")
    ap.add_argument("--known-len", type=int, default=10)
    ap.add_argument("--out-json", type=str, default="reports/qc_like_decrypt_benchmark.json")
    ap.add_argument("--out-md", type=str, default="reports/qc_like_decrypt_benchmark.md")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    rows = run_sweep(bits_list, args.known_len)
    payload = {
        "config": {"bits": bits_list, "known_len": int(args.known_len)},
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
