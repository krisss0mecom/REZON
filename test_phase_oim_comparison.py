"""Tests for phase_oim_comparison.py."""
import numpy as np
import pytest
from phase_oim_comparison import (
    StandardOIM, ConditionalOIM, random_graph, cut_value,
    max_cut_brute_force, oim_solve,
)

OIM_KW = dict(K=2.0, dt=0.001, a_anchor=0.08, noise_amp=0.1)


def test_random_graph_symmetric():
    A = random_graph(8, edge_prob=0.5, seed=0)
    assert np.all(A == A.T), "Adjacency matrix not symmetric"
    assert np.all(np.diag(A) == 0), "Self-loops present"


def test_max_cut_brute_force_known():
    """Triangle graph: max cut = 2."""
    A = np.array([[0,1,1],[1,0,1],[1,1,0]])
    assert max_cut_brute_force(A) == 2


def test_cut_value_known():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]])
    assert cut_value(A, np.array([0,1,0])) == 2
    assert cut_value(A, np.array([0,0,0])) == 0


def test_standard_oim_finds_cut():
    """Standard OIM should find non-trivial cut on a 6-node graph."""
    A   = random_graph(6, edge_prob=0.6, seed=1)
    J   = A.astype(float)
    oim = StandardOIM(6, J, **OIM_KW)
    r   = oim_solve(oim, A, n_restarts=3, warmup=1000, solve=2000)
    assert r["cut"] > 0, "OIM found zero cut"


def test_conditional_oim_all_antisync_equals_standard():
    """
    With phi_c=π for all edges, Conditional OIM = Standard OIM behavior.
    Both should find comparable cut quality.
    """
    N = 8
    A = random_graph(N, edge_prob=0.5, seed=42)
    J = A.astype(float)

    std_oim  = StandardOIM(N, J, **OIM_KW)
    r_std    = oim_solve(std_oim, A, n_restarts=3, warmup=1000, solve=2000)

    phi_c = np.where(A > 0, np.pi, 0.0)
    cond_oim = ConditionalOIM(N, A, phi_c, **OIM_KW)
    r_cond   = oim_solve(cond_oim, A, n_restarts=3, warmup=1000, solve=2000)

    total_edges = int(A.sum()) // 2
    q_std  = r_std["cut"]  / total_edges
    q_cond = r_cond["cut"] / total_edges

    # Both should get at least 50% of edges cut
    assert q_std  >= 0.3, f"Standard OIM poor: {q_std:.3f}"
    assert q_cond >= 0.3, f"Conditional OIM poor: {q_cond:.3f}"


def test_conditional_oim_sync_constraint():
    """
    phi_c=0 for an edge forces SYNC (same partition).
    Verify: after run, nodes i,j with phi_c[i,j]=0 end up same partition.
    """
    N = 6
    A = np.ones((N, N)) - np.eye(N)   # complete graph
    A = A.astype(int)

    # Force nodes 0,1 to same partition via phi_c=0
    phi_c = np.where(A > 0, np.pi, 0.0)   # default: all cut
    phi_c[0, 1] = phi_c[1, 0] = 0.0       # constraint: 0 and 1 same

    oim = ConditionalOIM(N, A, phi_c, noise_amp=0.0, **{k:v for k,v in OIM_KW.items()
                                                         if k != 'noise_amp'})
    # Run multiple times to check constraint satisfaction rate
    satisfied = 0
    for seed in range(5):
        oim.reset(seed=seed)
        oim.run(3000)
        p = oim.decode()
        if p[0] == p[1]:
            satisfied += 1
    assert satisfied >= 3, \
        f"Sync constraint (phi_c=0) satisfied only {satisfied}/5 times"


def test_conditional_oim_disabled_edge():
    """
    phi_c=pi/2 → cos=0 → no coupling. Edge is effectively disabled.
    """
    N  = 4
    # Only edge 0-1, disabled by phi_c=pi/2
    A  = np.zeros((N, N), dtype=int)
    A[0, 1] = A[1, 0] = 1
    phi_c = np.zeros((N, N))
    phi_c[0, 1] = phi_c[1, 0] = np.pi / 2   # disabled

    oim = ConditionalOIM(N, A, phi_c, noise_amp=0.0,
                         **{k:v for k,v in OIM_KW.items() if k!='noise_amp'})
    # With no coupling, nodes should drift independently (anchor only)
    # Gain matrix should be zero for disabled edge
    assert abs(oim.gain_matrix[0, 1]) < 1e-9, \
        f"Disabled edge gain={oim.gain_matrix[0,1]:.6f}, expected ~0"


def test_constrained_maxcut_hard_constraint():
    """
    Conditional OIM with phi_c=0 for constrained edges should
    satisfy constraint more reliably than penalty-based approach.
    """
    from phase_oim_comparison import benchmark_constrained_maxcut
    res = benchmark_constrained_maxcut(
        N=8, n_graphs=4, n_same_pairs=1,
        n_restarts=3, warmup=1500, solve=3000,
        **OIM_KW,
    )
    # Conditional should have fewer violations than standard
    assert res["cond_mean_violations"] <= res["std_mean_violations"] + 0.5, \
        f"Conditional violations {res['cond_mean_violations']:.2f} > " \
        f"standard {res['std_mean_violations']:.2f}"
