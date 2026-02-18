#!/usr/bin/env python3
import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np


def keystream_byte(key: int, idx: int) -> int:
    x = (key * 0x9E3779B185EBCA87 + (idx + 1) * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    return x & 0xFF


def xor_encrypt(blob: bytes, key: int) -> bytes:
    return bytes([b ^ keystream_byte(key, i) for i, b in enumerate(blob)])


def make_test_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "REZON synthetic encrypted-folder benchmark\n")
        zf.writestr("data/values.csv", "id,value\n1,42\n2,1337\n3,2026\n")
        zf.writestr("notes/info.txt", "This is synthetic data for RC benchmark only.\n")
    return buf.getvalue()


def looks_like_zip_header(dec: bytes) -> bool:
    return len(dec) >= 4 and dec[0:4] == b"PK\x03\x04"


def validate_full_zip(dec: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(dec), "r") as zf:
            names = zf.namelist()
            if not names:
                return False
            _ = zf.read(names[0])
        return True
    except Exception:
        return False


def brute_force_recover(ct: bytes, bits: int):
    attempts = 0
    keyspace = 1 << bits
    for k in range(keyspace):
        attempts += 1
        dec4 = bytes([ct[i] ^ keystream_byte(k, i) for i in range(4)])
        if dec4 != b"PK\x03\x04":
            continue
        dec = xor_encrypt(ct, k)
        if validate_full_zip(dec):
            return k, attempts
    return None, attempts


def phase_guided_score(key: int, ct: bytes, known_prefix: bytes) -> float:
    phi_a = (key & 0xFF) / 255.0 * 2.0 * np.pi
    phi_b = ((key >> 8) & 0xFF) / 255.0 * 2.0 * np.pi
    coupling = 1.9
    leak = 0.02
    dt = 0.015
    anchor_hz = 200.0
    anchor_amp = 0.4

    err = 0.0
    for i, b in enumerate(known_prefix):
        pred = ct[i] ^ keystream_byte(key, i)
        err += abs(pred - b) / 255.0
    err /= max(1, len(known_prefix))

    t = 0.0
    for _ in range(8):
        d = np.sin(phi_b - phi_a)
        drive = (1.0 - err) * 1.5
        dphi_a = coupling * d + anchor_amp * np.sin(2.0 * np.pi * anchor_hz * t - phi_a) - leak * phi_a
        dphi_b = -coupling * d + drive * np.cos(phi_b) + anchor_amp * np.sin(2.0 * np.pi * anchor_hz * t - phi_b) - leak * phi_b
        phi_a = (phi_a + dt * dphi_a) % (2.0 * np.pi)
        phi_b = (phi_b + dt * dphi_b) % (2.0 * np.pi)
        t += dt

    coherence = abs(np.cos(phi_a - phi_b))
    return float(err + 0.15 * coherence)


def phase_guided_recover(ct: bytes, bits: int, top_k: int):
    keyspace = 1 << bits
    prefix = b"PK\x03\x04"

    scores = np.empty(keyspace, dtype=np.float64)
    for k in range(keyspace):
        scores[k] = phase_guided_score(k, ct, prefix)
    order = np.argsort(scores)
    top = order[: min(top_k, keyspace)]

    attempts = 0
    for k in top:
        attempts += 1
        dec4 = bytes([ct[i] ^ keystream_byte(int(k), i) for i in range(4)])
        if dec4 != prefix:
            continue
        dec = xor_encrypt(ct, int(k))
        if validate_full_zip(dec):
            return int(k), attempts, True

    # fallback: continue full search after top-k
    tried = set(int(k) for k in top.tolist())
    for k in order[min(top_k, keyspace):]:
        kk = int(k)
        if kk in tried:
            continue
        attempts += 1
        dec4 = bytes([ct[i] ^ keystream_byte(kk, i) for i in range(4)])
        if dec4 != prefix:
            continue
        dec = xor_encrypt(ct, kk)
        if validate_full_zip(dec):
            return kk, attempts, False
    return None, attempts, False


def quantile_keys(keyspace: int, reps: int):
    if reps <= 1:
        qs = [0.5]
    elif reps == 2:
        qs = [0.30, 0.70]
    elif reps == 3:
        qs = [0.20, 0.50, 0.80]
    else:
        qs = [(i + 1) / (reps + 1) for i in range(reps)]
    return [min(keyspace - 1, max(0, int(q * keyspace))) for q in qs]


def run(bits_list, reps: int, top_k: int, out_dir: Path):
    base_zip = make_test_zip_bytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for bits in bits_list:
        keyspace = 1 << bits
        keys = quantile_keys(keyspace, reps)
        bf_attempts, bf_time = [], []
        pg_attempts, pg_time = [], []
        pg_topk_hits = 0
        sample_path = out_dir / f"sample_bits{bits}.enc"

        for idx, true_key in enumerate(keys):
            ct = xor_encrypt(base_zip, true_key)
            if idx == 0:
                sample_path.write_bytes(ct)

            t0 = time.perf_counter()
            k_bf, a_bf = brute_force_recover(ct, bits)
            bf_time.append(time.perf_counter() - t0)
            bf_attempts.append(a_bf)

            t1 = time.perf_counter()
            k_pg, a_pg, hit_topk = phase_guided_recover(ct, bits, top_k=top_k)
            pg_time.append(time.perf_counter() - t1)
            pg_attempts.append(a_pg)
            pg_topk_hits += int(hit_topk)

            if k_bf != true_key or k_pg != true_key:
                raise RuntimeError(f"recovery failed for bits={bits}, key={true_key}")

        bf_avg = float(np.mean(bf_attempts))
        pg_avg = float(np.mean(pg_attempts))
        rows.append(
            {
                "bits": bits,
                "keyspace": keyspace,
                "reps": reps,
                "keys_tested": keys,
                "sample_encrypted_file": str(sample_path),
                "bruteforce_avg_attempts": bf_avg,
                "phase_guided_avg_attempts": pg_avg,
                "attempt_gain_x": float(bf_avg / max(pg_avg, 1e-12)),
                "bruteforce_avg_time_s": float(np.mean(bf_time)),
                "phase_guided_avg_time_s": float(np.mean(pg_time)),
                "time_gain_x": float(np.mean(bf_time) / max(np.mean(pg_time), 1e-12)),
                "phase_topk_hits": int(pg_topk_hits),
                "phase_topk_hit_rate": float(pg_topk_hits / max(1, reps)),
                "grover_ideal_queries": float(math.sqrt(keyspace)),
            }
        )
    return rows


def to_md(rows):
    lines = [
        "# Synthetic Encrypted Folder Benchmark",
        "",
        "| bits | bf avg attempts | phase-guided avg attempts | attempt gain | bf avg time (s) | phase avg time (s) | time gain | top-k hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bits']} | {r['bruteforce_avg_attempts']:.2f} | {r['phase_guided_avg_attempts']:.2f} | "
            f"{r['attempt_gain_x']:.3f} | {r['bruteforce_avg_time_s']:.6f} | {r['phase_guided_avg_time_s']:.6f} | "
            f"{r['time_gain_x']:.3f} | {r['phase_topk_hit_rate']:.3f} |"
        )
    lines.append("")
    lines.append("Scope: synthetic benchmark only (controlled toy cipher + synthetic archive).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Synthetic encrypted-folder recovery benchmark: brute-force vs phase-guided.")
    ap.add_argument("--bits", type=str, default="8,10,12,14,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--out-json", type=str, default="reports/encrypted_folder_sim_benchmark.json")
    ap.add_argument("--out-md", type=str, default="reports/encrypted_folder_sim_benchmark.md")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    sample_dir = Path(args.out_json).parent / "encrypted_samples"
    rows = run(bits_list, reps=args.reps, top_k=args.top_k, out_dir=sample_dir)
    payload = {
        "config": {
            "bits": bits_list,
            "reps": int(args.reps),
            "top_k": int(args.top_k),
            "sample_dir": str(sample_dir),
            "note": "synthetic controlled benchmark only",
        },
        "rows": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_md(rows), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
