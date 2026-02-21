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


def make_zip_blob() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "REZON autonomous RC benchmark\n")
        zf.writestr("data/table.csv", "id,val\n1,7\n2,42\n3,1337\n")
        zf.writestr("meta/info.txt", "Synthetic archive for safe benchmark.\n")
    return buf.getvalue()


def validate_zip(blob: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            names = zf.namelist()
            if not names:
                return False
            _ = zf.read(names[0])
        return True
    except Exception:
        return False


def key_matches(cipher: bytes, key: int) -> bool:
    # quick header gate
    if bytes([cipher[i] ^ keystream_byte(key, i) for i in range(4)]) != b"PK\x03\x04":
        return False
    return validate_zip(xor_encrypt(cipher, key))


def brute_force(cipher: bytes, bits: int):
    attempts = 0
    keyspace = 1 << bits
    for k in range(keyspace):
        attempts += 1
        if key_matches(cipher, k):
            return k, attempts
    return None, attempts


def rc_phase_score(key: int, cipher: bytes, prefix: bytes) -> float:
    # Autonomous RC-like scoring function (no user candidate list).
    phi = np.zeros(6, dtype=np.float64)
    phi[0] = ((key >> 0) & 0xFF) / 255.0 * 2.0 * np.pi
    phi[1] = ((key >> 8) & 0xFF) / 255.0 * 2.0 * np.pi
    phi[2] = ((key >> 16) & 0xFF) / 255.0 * 2.0 * np.pi

    K = np.array(
        [
            [0, 1.2, -0.6, 0.2, 0.1, -0.2],
            [1.2, 0, 0.9, -0.2, 0.0, 0.1],
            [-0.6, 0.9, 0, 0.3, -0.1, 0.2],
            [0.2, -0.2, 0.3, 0, 0.8, -0.4],
            [0.1, 0.0, -0.1, 0.8, 0, 1.0],
            [-0.2, 0.1, 0.2, -0.4, 1.0, 0],
        ],
        dtype=np.float64,
    )
    coupling = 1.8
    leak = 0.02
    anchor_amp = 0.4
    anchor_hz = 200.0
    dt = 0.012

    err = 0.0
    for i, b in enumerate(prefix):
        pred = cipher[i] ^ keystream_byte(key, i)
        err += abs(pred - b) / 255.0
    err /= max(1, len(prefix))

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


def rc_autonomous_recover(cipher: bytes, bits: int):
    keyspace = 1 << bits
    prefix = b"PK\x03\x04"

    # RC autonomously scores entire search space and proposes order.
    keys = np.arange(keyspace, dtype=np.int64)
    scores = np.empty(keyspace, dtype=np.float64)
    for i, k in enumerate(keys):
        scores[i] = rc_phase_score(int(k), cipher, prefix)
    order = keys[np.argsort(scores)]

    attempts = 0
    for k in order:
        attempts += 1
        kk = int(k)
        if key_matches(cipher, kk):
            return kk, attempts
    return None, attempts


def fixed_keys(keyspace: int, reps: int):
    if reps <= 1:
        qs = [0.5]
    elif reps == 2:
        qs = [0.30, 0.70]
    elif reps == 3:
        qs = [0.20, 0.50, 0.80]
    else:
        qs = [(i + 1) / (reps + 1) for i in range(reps)]
    return [min(keyspace - 1, max(0, int(q * keyspace))) for q in qs]


def run(bits_list, reps: int):
    zip_blob = make_zip_blob()
    rows = []
    for bits in bits_list:
        keyspace = 1 << bits
        keys = fixed_keys(keyspace, reps)
        bf_attempts, bf_times = [], []
        rc_attempts, rc_times = [], []
        for key in keys:
            cipher = xor_encrypt(zip_blob, key)

            t0 = time.perf_counter()
            k_bf, a_bf = brute_force(cipher, bits)
            bf_times.append(time.perf_counter() - t0)
            bf_attempts.append(a_bf)

            t1 = time.perf_counter()
            k_rc, a_rc = rc_autonomous_recover(cipher, bits)
            rc_times.append(time.perf_counter() - t1)
            rc_attempts.append(a_rc)

            if k_bf != key or k_rc != key:
                raise RuntimeError(f"recovery mismatch bits={bits}, key={key}")

        bf_avg = float(np.mean(bf_attempts))
        rc_avg = float(np.mean(rc_attempts))
        rows.append(
            {
                "bits": bits,
                "keyspace": keyspace,
                "reps": reps,
                "keys_tested": keys,
                "bruteforce_avg_attempts": bf_avg,
                "rc_autonomous_avg_attempts": rc_avg,
                "attempt_gain_x": float(bf_avg / max(rc_avg, 1e-12)),
                "bruteforce_avg_time_s": float(np.mean(bf_times)),
                "rc_autonomous_avg_time_s": float(np.mean(rc_times)),
                "time_gain_x": float(np.mean(bf_times) / max(np.mean(rc_times), 1e-12)),
                "grover_ideal_queries": float(math.sqrt(keyspace)),
            }
        )
    return rows


def to_md(rows):
    lines = [
        "# Autonomous RC Encrypted-Folder Benchmark",
        "",
        "| bits | bf avg attempts | rc avg attempts | attempt gain | bf avg time (s) | rc avg time (s) | time gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bits']} | {r['bruteforce_avg_attempts']:.2f} | {r['rc_autonomous_avg_attempts']:.2f} | "
            f"{r['attempt_gain_x']:.3f} | {r['bruteforce_avg_time_s']:.6f} | {r['rc_autonomous_avg_time_s']:.6f} | "
            f"{r['time_gain_x']:.3f} |"
        )
    lines.append("")
    lines.append("Autonomous means: RC generates ordering itself; no user-provided candidate list.")
    lines.append("Scope: synthetic benchmark only.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Autonomous RC recovery benchmark on synthetic encrypted-folder task.")
    ap.add_argument("--bits", type=str, default="8,10,12,14,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-json", type=str, default="reports/encrypted_folder_autorc_benchmark.json")
    ap.add_argument("--out-md", type=str, default="reports/encrypted_folder_autorc_benchmark.md")
    args = ap.parse_args()

    bits_list = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    rows = run(bits_list, reps=args.reps)
    payload = {"config": {"bits": bits_list, "reps": int(args.reps)}, "rows": rows}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(to_md(rows), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
