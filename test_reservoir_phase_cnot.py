import numpy as np

from reservoir_phase_cnot import ReservoirPhaseCNOT


def _run(ctrl, targ, seed, steps=600):
    np.random.seed(seed)
    rc = ReservoirPhaseCNOT(noise_amp=0.0)
    rc.reset()
    for _ in range(steps):
        rc.step(ctrl, targ)
    _, bits = rc.readout()
    return int(bits[1])


def test_cnot_truth_table_many_seeds():
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for seed in range(50):
        for ctrl, targ in inputs:
            pred_t = _run(ctrl, targ, seed)
            expected_t = ctrl ^ targ
            assert pred_t == expected_t, (
                f"seed={seed} ctrl={ctrl} targ={targ} pred={pred_t} expected={expected_t}"
            )
