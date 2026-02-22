#!/usr/bin/env python3
"""
Phase Full Adder — binary arithmetic from cascaded phase gates.

════════════════════════════════════════════════════════════════════════
1-BIT FULL ADDER:  5 cascaded phase gates, no RLS, no trained weights.

  P     = XOR(A, B)          phase_gate: K·cos(φ_A)·sin(φ_B − φ_P)
  SUM   = XOR(P, Cin)        same equation, P output feeds as input
  G     = AND(A, B)          carry generate
  T     = AND(P, Cin)        carry propagate
  CARRY = OR(G, T)           carry output

  Output phases propagate directly between stages — no quantization.
  Cascade: φ_out of stage N is injection target of stage N+1.

════════════════════════════════════════════════════════════════════════
4-BIT RIPPLE CARRY ADDER:  4 full adders in sequence.

  Carry phase (φ_carry) flows from bit-0 to bit-3.
  No digital latching needed: phase continuity handles it.

════════════════════════════════════════════════════════════════════════
Anchor: 200 Hz (immutable REZON constraint).
"""

import argparse
import json
import os
import time

import numpy as np

from phase_gate_universal import ANCHOR_HZ, ensure_anchor_immutable

ensure_anchor_immutable(ANCHOR_HZ)

# ── Gate with arbitrary phase inputs ─────────────────────────────────────────

def run_gate_phase(
    gate_type: str,
    phi_c: float,
    phi_t: float,
    *,
    warmup: int = 2000,
    collect: int = 400,
    dt: float = 0.001,
    a_anchor: float = 0.08,
    K_inj: float = 8.0,
    K_main: float = 5.0,
    K_bias: float = 1.5,
    noise_amp: float = 0.0,
    run_seed: int = 0,
) -> float:
    """
    Evaluate a phase gate with continuous (analog) input phases.

    phi_c, phi_t : input phases in radians — any value, not just 0/π.
    Returns      : equilibrium output phase in [0, π] (arccos of mean cos).

    The output phase can be fed directly as input to the next gate,
    enabling arbitrary cascade depth without digital conversion.
    """
    rng = np.random.default_rng(run_seed)

    # Driven input oscillators start at their target phases
    phi_dc = float(phi_c)
    phi_dt = float(phi_t)

    # Output oscillator: start at π/2 (neutral point, equidistant from attractors {0,π}).
    # This avoids saddle-point trapping that occurs with random init near unstable FPs.
    phi_out = np.pi / 2.0
    omega = float(rng.uniform(-0.05, 0.05))
    t = 0.0
    cos_acc = 0.0

    for step_i in range(warmup + collect):
        # ── Advance driven inputs toward their target phases ───────────────
        for _ in range(1):
            anc_c = a_anchor * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi_dc)
            phi_dc = (phi_dc + dt * (K_inj * np.sin(phi_c - phi_dc) + anc_c)) % (2 * np.pi)

            anc_t = a_anchor * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi_dt)
            phi_dt = (phi_dt + dt * (K_inj * np.sin(phi_t - phi_dt) + anc_t)) % (2 * np.pi)

        # ── Gate equation ─────────────────────────────────────────────────
        if gate_type == "NOT":
            dphi = -K_main * np.sin(phi_dt - phi_out)

        elif gate_type == "XOR":
            dphi = K_main * np.cos(phi_dc) * np.sin(phi_dt - phi_out)

        elif gate_type == "AND":
            wire_gain = (1.0 - np.cos(phi_dc)) / 2.0
            dphi = -K_bias * np.sin(phi_out) + K_main * wire_gain * np.sin(phi_dt - phi_out)

        elif gate_type == "OR":
            wire_gain = (1.0 + np.cos(phi_dc)) / 2.0
            dphi = K_bias * np.sin(phi_out) + K_main * wire_gain * np.sin(phi_dt - phi_out)

        else:
            raise ValueError(f"Unknown gate: {gate_type!r}")

        # ── Advance output oscillator ──────────────────────────────────────
        anc_out = a_anchor * np.sin(2.0 * np.pi * ANCHOR_HZ * t - phi_out)
        noise = rng.normal(0.0, noise_amp) if noise_amp > 0 else 0.0
        phi_out = (phi_out + dt * (omega + anc_out + dphi + noise)) % (2 * np.pi)
        t += dt

        if step_i >= warmup:
            cos_acc += np.cos(phi_out)

    # Encode as phase in [0, π]: arccos(mean_cos)
    mean_cos = cos_acc / float(collect)
    return float(np.arccos(np.clip(mean_cos, -1.0, 1.0)))


def phase_to_bit(phi: float) -> int:
    """Decode phase → bit.  cos(φ) > 0 → 0,  cos(φ) < 0 → 1."""
    return 0 if np.cos(phi) > 0.0 else 1


# ── 1-bit Full Adder ──────────────────────────────────────────────────────────

def full_adder_1bit(
    a: int,
    b: int,
    cin: int,
    *,
    cin_phi: float = None,
    run_seed: int = 0,
    **gate_kwargs,
) -> dict:
    """
    1-bit full adder: SUM = A⊕B⊕Cin,  CARRY = AB + (A⊕B)·Cin

    Implemented as 5 cascaded phase gates:
      Stage 1: P     = XOR(A, B)
      Stage 2: SUM   = XOR(P, Cin)
      Stage 3: G     = AND(A, B)
      Stage 4: T     = AND(P, Cin)
      Stage 5: CARRY = OR(G, T)

    cin_phi: if set, use this phase for Cin (carry from previous adder stage).
             Enables continuous phase carry propagation without re-encoding.
    """
    phi_a   = float(a)   * np.pi
    phi_b   = float(b)   * np.pi
    phi_cin = float(cin_phi) if cin_phi is not None else float(cin) * np.pi

    # Stage 1: P = XOR(A, B)
    phi_p = run_gate_phase("XOR", phi_a, phi_b,
                           run_seed=run_seed + 0, **gate_kwargs)

    # Stage 2: SUM = XOR(P, Cin)  ← P phase feeds directly
    phi_sum = run_gate_phase("XOR", phi_p, phi_cin,
                             run_seed=run_seed + 1, **gate_kwargs)

    # Stage 3: G = AND(A, B)  [carry generate]
    phi_g = run_gate_phase("AND", phi_a, phi_b,
                           run_seed=run_seed + 2, **gate_kwargs)

    # Stage 4: T = AND(P, Cin)  [carry propagate]  ← P phase feeds directly
    phi_t = run_gate_phase("AND", phi_p, phi_cin,
                           run_seed=run_seed + 3, **gate_kwargs)

    # Stage 5: CARRY = OR(G, T)
    phi_carry = run_gate_phase("OR", phi_g, phi_t,
                               run_seed=run_seed + 4, **gate_kwargs)

    sum_bit   = phase_to_bit(phi_sum)
    carry_bit = phase_to_bit(phi_carry)

    expected_sum   = (a + b + cin) % 2
    expected_carry = (a + b + cin) // 2

    return {
        "a": a, "b": b, "cin": cin,
        "sum":   sum_bit,   "carry":   carry_bit,
        "phi_sum":   round(phi_sum,   4),
        "phi_carry": round(phi_carry, 4),
        "expected_sum":   expected_sum,
        "expected_carry": expected_carry,
        "sum_ok":   int(sum_bit   == expected_sum),
        "carry_ok": int(carry_bit == expected_carry),
        "ok":       int(sum_bit   == expected_sum and
                        carry_bit == expected_carry),
    }


def test_full_adder_1bit(**gate_kwargs) -> dict:
    """Test all 8 input combinations for 1-bit full adder."""
    combos = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    results, ok_count = [], 0
    for seed_off, (a, b, cin) in enumerate(combos):
        r = full_adder_1bit(a, b, cin, run_seed=seed_off * 13, **gate_kwargs)
        ok_count += r["ok"]
        results.append(r)
    return {"circuit": "full_adder_1bit",
            "score": ok_count, "total": len(combos), "rows": results}


# ── 4-bit Ripple Carry Adder ──────────────────────────────────────────────────

def ripple_carry_adder_4bit(a: int, b: int, *, run_seed: int = 0,
                             **gate_kwargs) -> dict:
    """
    4-bit ripple carry adder: a + b, a,b ∈ [0, 15].

    Carry phase propagates directly between stages (no re-encoding to bit).
    Result ∈ [0, 30] encoded as 4-bit sum + carry_out.
    """
    a_bits = [(a >> i) & 1 for i in range(4)]
    b_bits = [(b >> i) & 1 for i in range(4)]

    sum_bits  = []
    carry_bit = 0
    carry_phi = 0.0   # initial Cin = 0

    for i in range(4):
        r = full_adder_1bit(
            a_bits[i], b_bits[i], carry_bit,
            cin_phi=carry_phi,
            run_seed=run_seed + i * 20,
            **gate_kwargs,
        )
        sum_bits.append(r["sum"])
        carry_bit = r["carry"]
        carry_phi = r["phi_carry"]

    pred_total = sum(bit << i for i, bit in enumerate(sum_bits)) + (carry_bit << 4)
    expected   = a + b

    return {
        "a": a, "b": b,
        "sum_bits": sum_bits,
        "carry_out": carry_bit,
        "pred_total": pred_total,
        "expected":   expected,
        "ok": int(pred_total == expected),
    }


def test_ripple_carry_adder(n_cases: int = 20, seed: int = 7,
                             **gate_kwargs) -> dict:
    """Test n_cases random (a,b) pairs for 4-bit adder."""
    rng = np.random.default_rng(seed)
    pairs = [(int(rng.integers(0, 16)), int(rng.integers(0, 16)))
             for _ in range(n_cases)]
    results, ok_count = [], 0
    for run_i, (a, b) in enumerate(pairs):
        r = ripple_carry_adder_4bit(a, b, run_seed=run_i * 100, **gate_kwargs)
        ok_count += r["ok"]
        results.append(r)
    return {"circuit": "ripple_carry_4bit",
            "score": ok_count, "total": n_cases, "rows": results}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase Full Adder — cascaded gate arithmetic"
    )
    p.add_argument("--warmup",    type=int,   default=800)
    p.add_argument("--collect",   type=int,   default=200)
    p.add_argument("--dt",        type=float, default=0.001)
    p.add_argument("--K-inj",     type=float, default=8.0)
    p.add_argument("--K-main",    type=float, default=5.0)
    p.add_argument("--K-bias",    type=float, default=1.5)
    p.add_argument("--a-anchor",  type=float, default=0.08)
    p.add_argument("--noise",     type=float, default=0.0)
    p.add_argument("--adder-cases", type=int, default=20)
    p.add_argument("--out-json",  type=str,
                   default="reports/phase_full_adder_report.json")
    args = p.parse_args()

    gate_kw = dict(
        warmup=args.warmup, collect=args.collect, dt=args.dt,
        a_anchor=args.a_anchor, K_inj=args.K_inj,
        K_main=args.K_main, K_bias=args.K_bias, noise_amp=args.noise,
    )

    print("\n" + "═" * 64)
    print("  PHASE FULL ADDER — cascaded oscillator gates")
    print("  No RLS. No trained weights. Pure phase dynamics.")
    print("═" * 64)

    # ── 1-bit Full Adder ────────────────────────────────────────────────────
    print("\n[1] 1-bit Full Adder — all 8 input combinations")
    t0 = time.time()
    fa_res = test_full_adder_1bit(**gate_kw)
    elapsed = time.time() - t0
    print(f"    (a, b, cin) → (SUM, CARRY)")
    for r in fa_res["rows"]:
        sm = "✓" if r["sum_ok"]   else "✗"
        cm = "✓" if r["carry_ok"] else "✗"
        print(f"    ({r['a']},{r['b']},{r['cin']}) → "
              f"SUM[{sm}]={r['sum']} (exp {r['expected_sum']})  "
              f"CARRY[{cm}]={r['carry']} (exp {r['expected_carry']})")
    print(f"\n    Score: {fa_res['score']}/{fa_res['total']}  [{elapsed:.1f}s]")

    # ── 4-bit Ripple Carry Adder ─────────────────────────────────────────────
    print(f"\n[2] 4-bit Ripple Carry Adder — {args.adder_cases} random pairs")
    t0 = time.time()
    rc_res = test_ripple_carry_adder(n_cases=args.adder_cases, **gate_kw)
    elapsed = time.time() - t0
    for r in rc_res["rows"]:
        m = "✓" if r["ok"] else "✗"
        print(f"    [{m}] {r['a']:2d} + {r['b']:2d} = {r['pred_total']:2d} "
              f"(exp {r['expected']:2d})")
    print(f"\n    Score: {rc_res['score']}/{rc_res['total']}  [{elapsed:.1f}s]")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    fa_ok  = fa_res["score"] == fa_res["total"]
    rc_ok  = rc_res["score"] == rc_res["total"]
    print(f"  1-bit FA:      {fa_res['score']}/{fa_res['total']} "
          f"{'PASS ✓' if fa_ok else 'FAIL ✗'}")
    print(f"  4-bit Ripple:  {rc_res['score']}/{rc_res['total']} "
          f"{'PASS ✓' if rc_ok else 'FAIL ✗'}")
    print()
    print("  KEY RESULT: carry phase propagates between stages without")
    print("  quantization. Continuous phase cascade = analog carry chain.")
    print("═" * 64)

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "title": "Phase Full Adder — cascaded oscillator gates",
        "gate_params": gate_kw,
        "cascade_principle": (
            "Output phase φ_out of gate N is injection target of gate N+1. "
            "No digital encoding between stages. Phase continuity preserved."
        ),
        "full_adder_1bit":  fa_res,
        "ripple_carry_4bit": rc_res,
        "anchor_hz": ANCHOR_HZ,
        "ts": int(time.time()),
    }
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved → {args.out_json}")


if __name__ == "__main__":
    main()
