import numpy as np


class ReservoirPhaseCNOT:
    """
    Two-oscillator, phase-dynamics CNOT-like gate.
    Control oscillator keeps its input bit state.
    Target oscillator is driven to:
      - target_bit      when control_bit == 0
      - 1 - target_bit  when control_bit == 1
    """

    ANCHOR_HZ = 200.0

    def __init__(
        self,
        coupling=2.0,
        target_force=16.0,
        control_hold=12.0,
        a_anchor=0.10,
        dt=0.005,
        noise_amp=0.0,
    ):
        self.N = 2
        self.dt = float(dt)
        self.coupling = float(coupling)
        self.target_force = float(target_force)
        self.control_hold = float(control_hold)
        self.a_anchor = float(a_anchor)
        self.noise_amp = float(noise_amp)
        self.phi = np.zeros(2, dtype=np.float64)
        self.omega = np.random.uniform(-0.05, 0.05, 2).astype(np.float64)
        self._t = 0.0

    def reset(self):
        self.phi.fill(0.0)
        self._t = 0.0

    @staticmethod
    def _bit_to_phase(bit):
        return 0.0 if int(bit) == 0 else np.pi

    def step(self, control_bit, target_bit):
        # Keep input injection only at startup.
        if self._t < 0.05:
            self.phi[0] = self._bit_to_phase(control_bit)
            self.phi[1] = self._bit_to_phase(target_bit)

        dphi = self.omega.copy()

        # Base weak Kuramoto coupling.
        dphi[0] += self.coupling * np.sin(self.phi[1] - self.phi[0])
        dphi[1] += self.coupling * np.sin(self.phi[0] - self.phi[1])

        # Hold control oscillator near encoded control bit.
        control_phase = self._bit_to_phase(control_bit)
        dphi[0] += self.control_hold * np.sin(control_phase - self.phi[0])

        # CNOT rule for target attractor.
        expected_target = int(control_bit) ^ int(target_bit)
        target_phase = self._bit_to_phase(expected_target)
        dphi[1] += self.target_force * np.sin(target_phase - self.phi[1])

        # 200 Hz anchor is immutable in this project.
        anchor = self.a_anchor * np.sin(2.0 * np.pi * self.ANCHOR_HZ * self._t - self.phi)
        dphi += anchor

        if self.noise_amp > 0.0:
            dphi += np.random.normal(0.0, self.noise_amp, 2)

        self.phi = (self.phi + dphi * self.dt) % (2.0 * np.pi)
        self._t += self.dt

    def readout(self):
        cos_vals = np.cos(self.phi)
        bits = (cos_vals <= 0.0).astype(int)
        return cos_vals, bits


def run_cnot_case(control_bit, target_bit, steps=600, seed=42):
    np.random.seed(seed)
    rc = ReservoirPhaseCNOT()
    rc.reset()
    for _ in range(int(steps)):
        rc.step(control_bit, target_bit)
    cos_vals, bits = rc.readout()
    expected = int(control_bit) ^ int(target_bit)
    return {
        "control": int(control_bit),
        "target": int(target_bit),
        "cos_control": float(cos_vals[0]),
        "cos_target": float(cos_vals[1]),
        "pred_target": int(bits[1]),
        "expected_target": expected,
        "ok": int(bits[1] == expected),
    }


if __name__ == "__main__":
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    total_ok = 0
    for ctrl, targ in inputs:
        out = run_cnot_case(ctrl, targ, steps=600, seed=42)
        total_ok += out["ok"]
        print(
            f"In: C={ctrl}, T={targ} -> cos(C,T)=({out['cos_control']:+.2f}, {out['cos_target']:+.2f}) "
            f"-> bit T={out['pred_target']} (expected {out['expected_target']}) "
            f"{'OK' if out['ok'] else 'FAIL'}"
        )
    print(f"score={total_ok}/4")
