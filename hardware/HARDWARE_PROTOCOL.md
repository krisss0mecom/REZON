# Hardware Protocol (80 Oscillators)

Scope: compare hardware and software on identical instances and metrics.

## Setup
- 8 boards x 10 oscillators = 80 oscillators (simultaneous)
- Immutable anchor: 200.0 Hz
- Pipeline: Jetson -> AD9850 (anchor) -> RC boards -> AD7606 -> Jetson

## Mandatory logging fields
- `timestamp_utc`
- `instance_id`
- `problem` in {`maxcut`,`qubo`,`sat`}
- `seed`
- `method` in {`no_rc`,`rc_no_anchor`,`rc_anchor`,`rc_anchor_rls`,`hw_rc_anchor`,`hw_rc_anchor_rls`}
- `quality`
- `runtime_s`
- `power_w` (measured average)
- `energy_j` (`runtime_s * power_w`)
- `stability_tag` (optional: stable/unsteady)

## Run protocol
1. Generate fixed instance set in software with deterministic seeds.
2. Export instance files and load same set for hardware run.
3. Run each instance with same `seed` list for all methods.
4. Collect at least 8 seeds per instance.
5. Export CSV into `hardware/hw_runs.csv`.
6. Compare using `hardware/compare_hw_sw.py`.

## Claim constraints
- Allowed: "quantum-inspired analog search heuristic"
- Not allowed: "QC replacement"
