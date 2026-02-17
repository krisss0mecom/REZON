#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass

import numpy as np


ANCHOR_HZ = 200.0


def ensure_anchor_immutable(value: float) -> None:
    if abs(float(value) - ANCHOR_HZ) > 1e-6:
        raise ValueError(f"f_anchor must be exactly {ANCHOR_HZ} Hz")


@dataclass
class RCConfig:
    n: int = 24
    dt: float = 0.01
    coupling: float = 1.9
    a_anchor: float = 1.0
    leak: float = 0.01
    input_gain: float = 2.2
    noise_amp: float = 0.01
    seed: int = 42


class PhaseReservoirCNOT:
    """
    Neutral phase reservoir.
    Inputs are only control/target drives; XOR mapping is learned by RLS head.
    """

    def __init__(self, cfg: RCConfig):
        self.cfg = cfg
        ensure_anchor_immutable(ANCHOR_HZ)
        self.rng = np.random.default_rng(cfg.seed)
        self.n = int(cfg.n)
        self.dt = float(cfg.dt)
        self.coupling = float(cfg.coupling)
        self.a_anchor = float(cfg.a_anchor)
        self.leak = float(cfg.leak)
        self.input_gain = float(cfg.input_gain)
        self.noise_amp = float(cfg.noise_amp)
        self._t = 0.0
        self.phi = np.zeros(self.n, dtype=np.float64)
        self.omega = self.rng.uniform(-0.25, 0.25, size=self.n)
        k = self.rng.normal(0.0, 1.0, size=(self.n, self.n))
        k = 0.5 * (k + k.T)
        mask = self.rng.uniform(0.0, 1.0, size=(self.n, self.n)) < 0.70
        k[mask] = 0.0
        np.fill_diagonal(k, 0.0)
        spec = np.max(np.abs(np.linalg.eigvals(k))) + 1e-9
        self.k = (k / spec) * self.coupling
        self.reset()

    @staticmethod
    def _bit_to_phase(bit: int) -> float:
        return 0.0 if int(bit) == 0 else np.pi

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(self.cfg.seed if seed is None else int(seed))
        self.phi = rng.uniform(0.0, 2.0 * np.pi, size=self.n)
        self._t = 0.0

    def _inject_inputs(self, control_bit: int, target_bit: int) -> None:
        cp = self._bit_to_phase(control_bit)
        tp = self._bit_to_phase(target_bit)
        self.phi[0] = (self.phi[0] + self.input_gain * np.sin(cp - self.phi[0])) % (2.0 * np.pi)
        self.phi[1] = (self.phi[1] + self.input_gain * np.sin(tp - self.phi[1])) % (2.0 * np.pi)

    def step(self, control_bit: int, target_bit: int) -> None:
        dphi = self.omega.copy()
        # Vectorized Kuramoto interaction for speed.
        diff = self.phi[None, :] - self.phi[:, None]
        dphi += np.sum(self.k * np.sin(diff), axis=1)
        dphi += self.a_anchor * np.sin(2.0 * np.pi * ANCHOR_HZ * self._t - self.phi)
        dphi += -self.leak * self.phi
        if self.noise_amp > 0.0:
            dphi += self.rng.normal(0.0, self.noise_amp, size=self.n)
        self.phi = (self.phi + self.dt * dphi) % (2.0 * np.pi)
        self._inject_inputs(control_bit, target_bit)
        self._t += self.dt

    def features(self) -> np.ndarray:
        phi = self.phi
        base = np.concatenate(
            [
                np.sin(phi),
                np.cos(phi),
                np.sin(2.0 * phi),
                np.cos(2.0 * phi),
                phi / (2.0 * np.pi),
            ]
        ).astype(np.float64)
        # Nonlinear lifting so linear RLS can decode XOR-like structure.
        m = min(8, self.n)
        c = np.cos(phi[:m])
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append(c[i] * c[j])
        if pairs:
            base = np.concatenate([base, np.asarray(pairs, dtype=np.float64)])
        return base

    def collect_feature(self, control_bit: int, target_bit: int, warmup: int, collect: int) -> np.ndarray:
        for _ in range(max(0, int(warmup))):
            self.step(control_bit, target_bit)
        d = self.features().shape[0]
        acc = np.zeros(d, dtype=np.float64)
        c = max(1, int(collect))
        for _ in range(c):
            self.step(control_bit, target_bit)
            acc += self.features()
        return acc / float(c)


def rls_train_eval(
    cfg: RCConfig,
    warmup: int = 120,
    collect: int = 24,
    train_steps: int = 1600,
    rls_lambda: float = 0.995,
    p0: float = 20.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    rc = PhaseReservoirCNOT(cfg)
    d = rc.features().shape[0]
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    pdiag = np.full(d, float(p0), dtype=np.float64)
    eps = 1e-8
    combos = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int64)

    for step in range(int(train_steps)):
        idx = int(rng.integers(0, len(combos)))
        c, t = combos[idx]
        rc.reset(seed + 1000 + step * 17 + idx)
        x = rc.collect_feature(int(c), int(t), warmup=warmup, collect=collect)
        y_true = float(int(c) ^ int(t))
        y_pred = float(np.dot(x, w) + b)
        err = y_true - y_pred

        # Diagonal RLS
        px = pdiag * x
        denom = float(rls_lambda + np.dot(x, px) + eps)
        k = px / denom
        w += k * err
        b += 0.05 * err
        pdiag = (pdiag - k * x * pdiag) / rls_lambda
        pdiag = np.clip(pdiag, 1e-6, 1e6)

    details = []
    ok = 0
    for c, t in combos:
        rc.reset(seed + 99991 + 101 * int(c) + 1009 * int(t))
        x = rc.collect_feature(int(c), int(t), warmup=warmup, collect=collect)
        y_true = int(c) ^ int(t)
        y_pred = float(np.dot(x, w) + b)
        bit = int(y_pred >= 0.5)
        ok += int(bit == y_true)
        details.append(
            {
                "control": int(c),
                "target": int(t),
                "pred_score": y_pred,
                "pred_bit": bit,
                "expected_bit": y_true,
                "ok": int(bit == y_true),
            }
        )
    return {
        "score_4": int(ok),
        "accuracy": float(ok / 4.0),
        "details": details,
        "cfg": cfg.__dict__,
        "train_steps": int(train_steps),
        "warmup": int(warmup),
        "collect": int(collect),
        "rls_lambda": float(rls_lambda),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train RLS readout on phase reservoir for CNOT truth-table.")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--train-steps", type=int, default=1600)
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--collect", type=int, default=24)
    p.add_argument("--coupling", type=float, default=1.9)
    p.add_argument("--anchor", type=float, default=1.0)
    p.add_argument("--leak", type=float, default=0.01)
    p.add_argument("--input-gain", type=float, default=2.2)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--rls-lambda", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    cfg = RCConfig(
        n=args.n,
        coupling=args.coupling,
        a_anchor=args.anchor,
        leak=args.leak,
        input_gain=args.input_gain,
        noise_amp=args.noise,
        seed=args.seed,
    )
    out = rls_train_eval(
        cfg,
        warmup=args.warmup,
        collect=args.collect,
        train_steps=args.train_steps,
        rls_lambda=args.rls_lambda,
        seed=args.seed,
    )
    for row in out["details"]:
        print(
            f"In: C={row['control']}, T={row['target']} -> score={row['pred_score']:+.4f} "
            f"bit={row['pred_bit']} expected={row['expected_bit']} {'OK' if row['ok'] else 'FAIL'}"
        )
    print(f"score={out['score_4']}/4 acc={out['accuracy']:.4f}")
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
