# Statistical Power Notes (Pass-Rate Claims)

This project often reports pass rates for deterministic/noisy sweeps.
For binomial pass/fail outcomes, we use Wilson 95% confidence intervals.

## Why Seed Count Matters

If observed pass rate is 100% (`k=n`), the lower confidence bound depends on `n`:

| n seeds | Wilson 95% lower bound (k=n) |
|---:|---:|
| 12 | 0.7575 |
| 20 | 0.8389 |
| 30 | 0.8865 |
| 40 | 0.9124 |
| 60 | 0.9398 |
| 100 | 0.9630 |
| 200 | 0.9812 |

Interpretation:

- `n=12` is a smoke-level stability check.
- `n>=60` is a stronger claim (lower bound near 0.94).
- `n>=100` is preferred for high-confidence robustness statements.

## Reporting Policy

- Always publish `(n_pass / n_trials)` and Wilson CI.
- Avoid binary "robust/not robust" language without CI.
- For manuscript-level claims, target at least `n>=100` seeds on key sweeps.
