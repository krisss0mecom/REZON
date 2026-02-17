from cnot_rls import RCConfig, rls_train_eval


def test_cnot_rls_truth_table_many_seeds():
    cfg = RCConfig(n=24, coupling=1.9, a_anchor=1.0, leak=0.01, input_gain=2.2, noise_amp=0.005, seed=42)
    for seed in range(4):
        out = rls_train_eval(
            cfg,
            warmup=120,
            collect=24,
            train_steps=1600,
            rls_lambda=0.995,
            seed=seed,
        )
        assert out["score_4"] == 4, f"seed={seed} details={out['details']}"
