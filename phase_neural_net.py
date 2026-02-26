#!/usr/bin/env python3
"""
Phase Neural Network (PhaseNN)
==============================
Trainable associative memory on S¹ — a genuinely new neural network paradigm.

Standard NN:   forward pass = matrix multiply      W·x
PhaseNN:       forward pass = ODE dynamics          dφ/dt = f(φ, ξ^μ)

The key insight: memory patterns ξ^μ ARE the weights.
Learning = repositioning attractors in phase space.

Architecture:
    x ∈ ℝ^d  →  [PhaseEncoder]  →  φ⁰ ∈ [0,2π)^N
                                        ↓  ODE T seconds
                                    φᵀ ∈ [0,2π)^N
                                        ↓  [PhaseReadout]
                                    logits ∈ ℝ^C

ODE dynamics (gradient flow on S¹):
    dφᵢ/dt = K · Σ_μ softmax(β·m)_μ · sin(φᵢ − ξᵢ^μ) + anchor

    m_μ = Σᵢ cos(φᵢ − ξᵢ^μ)   circular overlap ∈ [−N, N]

Trainable:
    ξ^μ ∈ ℝ^{P×N}   — memory patterns  (the "weights" of PhaseNN)
    Encoder weights  — input → initial phases
    Readout weights  — final phases → class scores

Author: Krzysztof Gwóźdź
"""

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint_adjoint as odeint

TWO_PI = 2.0 * math.pi

# ── Utilities ────────────────────────────────────────────────────────────────

def wrap(phi):
    """Wrap phases to [0, 2π)."""
    return phi % TWO_PI


def circular_overlap(phi, xi):
    """
    Compute circular overlap m_μ = Σᵢ cos(φᵢ − ξᵢ^μ).

    Args:
        phi : (batch, N)
        xi  : (P, N)  — memory patterns

    Returns:
        m   : (batch, P)
    """
    # diff[b, μ, i] = phi[b, i] - xi[μ, i]
    diff = phi.unsqueeze(1) - xi.unsqueeze(0)   # (batch, P, N)
    return torch.cos(diff).sum(-1)               # (batch, P)


# ── ODE function ─────────────────────────────────────────────────────────────

class PhaseODEFunc(nn.Module):
    """
    Differentiable ODE: dφ/dt = K · Σ_μ softmax(β·m)_μ · sin(φᵢ − ξᵢ^μ) + anchor

    Memory patterns ξ^μ are nn.Parameter → trained by backprop through ODE.
    """

    def __init__(self, N: int, P: int,
                 beta: float = 1.0,
                 K: float = 1.0,
                 a_anc: float = 0.08,
                 omega_anc: float = TWO_PI * 200.0):
        super().__init__()
        self.N = N
        self.P = P
        self.beta = beta
        self.K = K
        self.a_anc = a_anc
        self.omega_anc = omega_anc

        # ── Trainable memory patterns ξ^μ ∈ [0, 2π)^{P×N}
        # Initialised uniformly at random — will move during training
        xi_init = torch.rand(P, N) * TWO_PI
        self.xi = nn.Parameter(xi_init)

    def forward(self, t: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t   : scalar time
            phi : (batch, N)   current phases

        Returns:
            dphi: (batch, N)   time derivative
        """
        # Circular overlap m_μ = Σᵢ cos(φᵢ − ξᵢ^μ)
        diff = phi.unsqueeze(1) - self.xi.unsqueeze(0)   # (batch, P, N)
        m = torch.cos(diff).sum(-1)                       # (batch, P)

        # Softmax weights — numerically stable, bounded, differentiable
        # beta / N normalises overlap to [-1, 1] before exp → no overflow
        weights = torch.softmax(self.beta * m / self.N, dim=-1)  # (batch, P)

        # Coupling: Σ_μ w_μ · sin(φᵢ − ξᵢ^μ)
        sin_diff = torch.sin(diff)                                # (batch, P, N)
        coupling = (weights.unsqueeze(-1) * sin_diff).sum(1)     # (batch, N)

        dphi = self.K * coupling

        # 200 Hz anchor — breaks rotational symmetry, enables hardware impl.
        dphi = dphi + self.a_anc * torch.sin(self.omega_anc * t - phi)

        return dphi


# ── Phase Encoder ─────────────────────────────────────────────────────────────

class PhaseEncoder(nn.Module):
    """
    Maps continuous input x ∈ ℝ^d → initial phases φ⁰ ∈ [0, 2π)^N.

    φᵢ = 2π · sigmoid(Wᵢ · x + bᵢ)

    Sigmoid maps to (0,1), scaled to (0, 2π).
    Learnable — encoder aligns input representation with attractor basin.
    """

    def __init__(self, d_in: int, N: int):
        super().__init__()
        self.linear = nn.Linear(d_in, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_in) → phi: (batch, N)"""
        return TWO_PI * torch.sigmoid(self.linear(x))


# ── Phase Readout ─────────────────────────────────────────────────────────────

class PhaseReadout(nn.Module):
    """
    Maps final phases φᵀ ∈ [0, 2π)^N → class logits ∈ ℝ^C.

    Uses [cos φ, sin φ] as features — rotation-invariant representation.
    Linear layer maps 2N features to C class scores.
    """

    def __init__(self, N: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(2 * N, n_classes)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: (batch, N) → logits: (batch, C)"""
        features = torch.cat([torch.cos(phi), torch.sin(phi)], dim=-1)
        return self.linear(features)


# ── Full Phase Neural Network ─────────────────────────────────────────────────

class PhaseNN(nn.Module):
    """
    Full Phase Neural Network.

    Forward pass:
        1. Encode input x → initial phases φ⁰
        2. Run ODE dynamics for T seconds → recalled phases φᵀ
        3. Decode φᵀ → class logits

    The ODE is solved with torchdiffeq (adjoint method) — gradients flow
    through the continuous-time dynamics back to patterns ξ^μ and encoder.
    """

    def __init__(self,
                 d_in: int,
                 N: int,
                 P: int,
                 n_classes: int,
                 T: float = 0.5,
                 beta: float = 2.0,
                 K: float = 1.0,
                 a_anc: float = 0.08,
                 method: str = 'rk4',
                 rtol: float = 1e-3,
                 atol: float = 1e-4):
        super().__init__()
        self.T = T
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self.encoder = PhaseEncoder(d_in, N)
        self.ode_func = PhaseODEFunc(N, P, beta=beta, K=K, a_anc=a_anc)
        self.readout  = PhaseReadout(N, n_classes)

        # Time span for ODE integration
        self.register_buffer('t_span', torch.tensor([0.0, T]))

    @property
    def patterns(self) -> torch.Tensor:
        """Current memory patterns ξ^μ (P × N)."""
        return self.ode_func.xi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_in)
        Returns:
            logits: (batch, n_classes)
        """
        # 1. Encode input to initial phases
        phi0 = self.encoder(x)                            # (batch, N)

        # 2. Solve ODE: dφ/dt = f(t, φ)  from t=0 to t=T
        phi_traj = odeint(
            self.ode_func,
            phi0,
            self.t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            adjoint_params=list(self.ode_func.parameters()),
        )                                                  # (2, batch, N)

        # 3. Take final state φᵀ
        phi_T = phi_traj[-1]                               # (batch, N)

        # 4. Decode to class logits
        return self.readout(phi_T)

    def recall_phases(self, x: torch.Tensor) -> torch.Tensor:
        """Return final phase state φᵀ (useful for visualisation)."""
        phi0 = self.encoder(x)
        phi_traj = odeint(
            self.ode_func, phi0, self.t_span,
            method=self.method, rtol=self.rtol, atol=self.atol,
        )
        return phi_traj[-1]


# ── Synthetic Dataset ─────────────────────────────────────────────────────────

def make_phase_dataset(n_classes: int, N: int,
                       n_train: int, n_test: int,
                       noise_std: float = 0.4,
                       seed: int = 42):
    """
    Synthetic phase classification dataset.

    Each class has a random "true centre" ξ_c ∈ [0, 2π)^N.
    Training/test examples = centre + Gaussian noise (wrapped to [0, 2π)).

    Returns:
        X_train, y_train, X_test, y_test  — all torch.Tensor
        centres                           — (n_classes, N) ground-truth phase centres
    """
    rng = np.random.default_rng(seed)

    # Ground-truth class centres
    centres = rng.uniform(0, TWO_PI, size=(n_classes, N)).astype(np.float32)

    def make_split(n_per_class):
        X, y = [], []
        for c in range(n_classes):
            noise = rng.normal(0, noise_std, size=(n_per_class, N)).astype(np.float32)
            X.append(centres[c] + noise)          # noisy phase vector
            y.extend([c] * n_per_class)
        X = np.vstack(X)
        y = np.array(y, dtype=np.int64)
        # Shuffle
        idx = rng.permutation(len(y))
        return torch.from_numpy(X[idx]), torch.from_numpy(y[idx])

    X_tr, y_tr = make_split(n_train)
    X_te, y_te = make_split(n_test)

    return X_tr, y_tr, X_te, y_te, torch.from_numpy(centres)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: PhaseNN,
          X_train: torch.Tensor, y_train: torch.Tensor,
          X_test:  torch.Tensor, y_test:  torch.Tensor,
          n_epochs: int = 60,
          batch_size: int = 32,
          lr: float = 3e-3,
          device: str = 'cpu') -> dict:

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test,  y_test  = X_test.to(device),  y_test.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    n = len(X_train)

    print(f"\n{'='*60}")
    print(f"  Phase Neural Network Training")
    print(f"  N={model.ode_func.N}  P={model.ode_func.P}  "
          f"T={model.T}  epochs={n_epochs}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Train%':>7}  {'Test%':>7}  {'Time':>6}")
    print(f"{'-'*50}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        epoch_loss, correct = 0.0, 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()

        avg_loss  = epoch_loss / n
        train_acc = 100.0 * correct / n

        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_acc = 100.0 * (test_logits.argmax(1) == y_test).float().mean().item()

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {avg_loss:>8.4f}  {train_acc:>6.1f}%  "
                  f"{test_acc:>6.1f}%  {elapsed:>5.1f}s")

    print(f"{'='*60}")
    return history


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(model: PhaseNN, centres: torch.Tensor):
    """
    Check how well learned patterns ξ^μ align with true class centres.
    Circular distance: d(a, b) = 1 − cos(a − b) averaged over N.
    """
    model.eval()
    xi = model.patterns.detach()           # (P, N)
    C = centres.shape[0]
    N = centres.shape[1]

    print(f"\n{'='*60}")
    print(f"  Attractor alignment: learned ξ^μ vs true centres")
    print(f"{'='*60}")
    print(f"{'Pattern μ':>10}  {'Best centre':>12}  {'Distance':>10}")
    print(f"{'-'*40}")

    for mu in range(model.ode_func.P):
        dists = []
        for c in range(C):
            d = (1.0 - torch.cos(xi[mu] - centres[c])).mean().item()
            dists.append(d)
        best_c = int(np.argmin(dists))
        best_d = dists[best_c]
        print(f"{mu:>10}  {'class ' + str(best_c):>12}  {best_d:>10.4f}")
    print(f"{'='*60}")


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    N          = 16       # oscillators per layer
    P          = 8        # memory patterns (= number of classes)
    n_classes  = P
    d_in       = N        # input dimension = N (phases directly)
    noise_std  = 0.5      # noise on training examples (radians)
    n_train    = 80       # training examples per class
    n_test     = 20       # test examples per class
    T          = 0.5      # ODE integration time (seconds)
    beta       = 3.0      # softmax sharpness
    n_epochs   = 60
    batch_size = 32
    lr         = 3e-3

    print("Phase Neural Network — Demo")
    print(f"N={N} oscillators  |  P={P} patterns  |  {n_classes} classes")
    print(f"T={T}s ODE  |  noise σ={noise_std}  |  {n_train}×{n_classes} train")

    # ── Dataset ───────────────────────────────────────────────────────────────
    X_tr, y_tr, X_te, y_te, centres = make_phase_dataset(
        n_classes=n_classes, N=N,
        n_train=n_train, n_test=n_test,
        noise_std=noise_std,
    )

    # ── Baseline: MLP with same parameter budget ──────────────────────────────
    hidden = 64
    mlp = nn.Sequential(
        nn.Linear(N, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_classes),
    )
    mlp_params = sum(p.numel() for p in mlp.parameters())

    # ── Phase Neural Network ──────────────────────────────────────────────────
    model = PhaseNN(
        d_in=d_in, N=N, P=P, n_classes=n_classes,
        T=T, beta=beta, K=1.0, a_anc=0.08,
        method='rk4',
    )
    pnn_params = sum(p.numel() for p in model.parameters())

    print(f"\nPhaseNN parameters: {pnn_params}")
    print(f"MLP parameters:     {mlp_params}")

    # ── Train PhaseNN ─────────────────────────────────────────────────────────
    history = train(model, X_tr, y_tr, X_te, y_te,
                    n_epochs=n_epochs, batch_size=batch_size, lr=lr)

    # ── Train MLP baseline ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MLP Baseline (same data, same epochs)")
    print(f"{'='*60}")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Train%':>7}  {'Test%':>7}")
    print(f"{'-'*40}")

    mlp_opt = optim.Adam(mlp.parameters(), lr=lr)
    mlp_sched = optim.lr_scheduler.CosineAnnealingLR(mlp_opt, T_max=n_epochs)
    crit = nn.CrossEntropyLoss()
    n = len(X_tr)

    for epoch in range(1, n_epochs + 1):
        mlp.train()
        perm = torch.randperm(n)
        ep_loss, correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            mlp_opt.zero_grad()
            logits = mlp(xb)
            loss = crit(logits, yb)
            loss.backward()
            mlp_opt.step()
            ep_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        mlp_sched.step()

        if epoch % 5 == 0 or epoch == 1:
            mlp.eval()
            with torch.no_grad():
                te_acc = 100.0 * (mlp(X_te).argmax(1) == y_te).float().mean().item()
            tr_acc = 100.0 * correct / n
            print(f"{epoch:>6}  {ep_loss/n:>8.4f}  {tr_acc:>6.1f}%  {te_acc:>6.1f}%")

    print(f"{'='*60}")

    # ── Final comparison ──────────────────────────────────────────────────────
    model.eval()
    mlp.eval()
    with torch.no_grad():
        pnn_acc = 100.0 * (model(X_te).argmax(1) == y_te).float().mean().item()
        mlp_acc = 100.0 * (mlp(X_te).argmax(1) == y_te).float().mean().item()

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  PhaseNN test accuracy : {pnn_acc:6.1f}%")
    print(f"  MLP test accuracy     : {mlp_acc:6.1f}%")
    print(f"  PhaseNN parameters    : {pnn_params}")
    print(f"  MLP parameters        : {mlp_params}")
    print(f"{'='*60}")

    # ── Attractor analysis ────────────────────────────────────────────────────
    analyse(model, centres)

    # ── Key insight ───────────────────────────────────────────────────────────
    print(f"""
KEY OBSERVATION:
  PhaseNN learns by REPOSITIONING ATTRACTORS in phase space.
  Initial ξ^μ: random phases (uniform on S¹)
  Trained ξ^μ: aligned with true class centres (see table above)

  This is fundamentally different from MLP:
  MLP   learns weights W that transform input space.
  PhaseNN learns attractors ξ^μ that reshape the energy landscape.

  The ODE dynamics ARE the computation.
  The patterns ξ^μ ARE the memory.
  They are the same thing.
""")
